#!/usr/bin/env/python

"""Recipe for training a transducer-based TS-ASR system
(see https://arxiv.org/abs/2209.04175).

To run this recipe, do the following:
> python train_librispeech_mix.py hparams/LibriSpeechMix/<config>.yaml

Authors
 * Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/recipes/LibriSpeech/ASR/transducer/train.py

# TODO: if_main_process during validation if problems occur in distributed mode
# TODO: check causality
# TODO: check where to inject embeddings
# TODO: train speaker encoder from scratch?

# Apply hotfix for SpeechBrain distributed execution
import distributed as hotfix
from speechbrain.utils import distributed

distributed.if_main_process = hotfix.if_main_process

import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main


class TSASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        mixed_wavs, mixed_wavs_lens = batch.mixed_sig
        enroll_wavs, enroll_wavs_lens = batch.enroll_sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add speed perturbation if specified
        if stage == sb.Stage.TRAIN:
            if "speed_perturb" in self.modules:
                mixed_wavs = self.modules.speed_perturb(mixed_wavs)

        # Extract speaker embedding (freeze speaker encoder)
        with torch.set_grad_enabled(not self.hparams.run_pretrainer):
            feats = self.modules.speaker_feature_extractor(enroll_wavs)
            feats = self.modules.speaker_normalizer(feats, enroll_wavs_lens)
            speaker_embs = self.modules.speaker_encoder(feats, enroll_wavs_lens)
        speaker_embs = self.modules.speaker_proj(speaker_embs)

        # Extract features
        feats = self.modules.feature_extractor(mixed_wavs)
        feats = self.modules.normalizer(feats, mixed_wavs_lens)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if "augmentation" in self.modules:
                feats = self.modules.augmentation(feats)

        # Forward encoder/transcriber
        feats = self.modules.cnn(feats)
        encoder_out = self.modules.encoder(feats, mixed_wavs_lens, speaker_embs)
        encoder_out = self.modules.encoder_proj(encoder_out)

        # Forward decoder/predictor
        embs = self.modules.embedding(tokens_bos)
        decoder_out, _ = self.modules.decoder(embs, lengths=tokens_bos_lens)
        decoder_out = self.modules.decoder_proj(decoder_out)

        # Forward joiner
        # Add label dimension to the encoder tensor: [B, T, H_enc] => [B, T, 1, H_enc]
        # Add time dimension to the decoder tensor: [B, U, H_dec] => [B, 1, U, H_dec]
        joiner_out = self.modules.joiner(
            encoder_out[..., None, :], decoder_out[:, None, ...]
        )

        # Compute transducer log-probabilities
        transducer_logits = self.modules.transducer_head(joiner_out)

        # Compute outputs
        hyps = None
        ctc_logprobs = None
        ce_logprobs = None

        if stage == sb.Stage.TRAIN:
            if (
                hasattr(self.hparams, "ctc_cost")
                and self.hparams.epoch_counter.current <= self.hparams.num_ctc_epochs
            ):
                # Output layer for CTC log-probabilities
                ctc_logits = self.modules.encoder_head(encoder_out)
                ctc_logprobs = ctc_logits.log_softmax(dim=-1)
            if (
                hasattr(self.hparams, "ce_cost")
                and self.hparams.epoch_counter.current <= self.hparams.num_ce_epochs
            ):
                # Output layer for CE log-probabilities
                ce_logits = self.modules.decoder_head(decoder_out)
                ce_logprobs = ce_logits.log_softmax(dim=-1)
        elif stage == sb.Stage.VALID:
            # During validation, run decoding only every valid_search_freq epochs to speed up training
            if self.hparams.epoch_counter.current % self.hparams.valid_search_freq == 0:
                hyps, scores, _, _ = self.hparams.greedy_searcher(encoder_out)
        else:
            hyps, scores, _, _ = self.hparams.beam_searcher(encoder_out)

        return transducer_logits, ctc_logprobs, ce_logprobs, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the transducer loss + (CTC + CE) given predictions and targets."""
        transducer_logits, ctc_logprobs, ce_logprobs, hyps = predictions

        ids = batch.id
        _, mixed_wavs_lens = batch.mixed_sig
        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = self.hparams.transducer_loss(
            transducer_logits, tokens, mixed_wavs_lens, tokens_lens
        )
        if ctc_logprobs is not None:
            loss += self.hparams.ctc_weight * self.hparams.ctc_loss(
                ctc_logprobs, tokens, mixed_wavs_lens, tokens_lens
            )
        if ce_logprobs is not None:
            loss += self.hparams.ce_weight * self.hparams.ce_loss(
                ce_logprobs, tokens_eos, length=tokens_eos_lens
            )

        if hyps is not None:
            target_words = batch.target_words

            # Decode predicted tokens to words
            predicted_words = self.tokenizer(hyps, task="decode_from_list")

            self.cer_metric.append(ids, predicted_words, target_words)
            self.wer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        with self.no_sync(not should_step):
            # Managing automatic mixed precision
            if self.auto_mix_prec:
                with torch.autocast(torch.device(self.device).type):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)

                # Losses are excluded from mixed precision to avoid instabilities
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

                self.scaler.scale(loss / self.grad_accumulation_factor).backward()

                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.check_gradients(loss):
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    self.hparams.lr_annealing(self.optimizer)
            else:
                if self.bfloat16_mix_prec:
                    with torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=torch.bfloat16,
                    ):
                        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
                else:
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
                (loss / self.grad_accumulation_factor).backward()
                if should_step:
                    if self.check_gradients(loss):
                        self.optimizer.step()
                    self.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    self.hparams.lr_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            if self.hparams.epoch_counter.current % self.hparams.valid_search_freq == 0:
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing.current_lr
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.hparams.epoch_counter.current % self.hparams.valid_search_freq == 0:
                if if_main_process():
                    self.checkpointer.save_and_keep_only(
                        meta={"WER": stage_stats["WER"]},
                        min_keys=["WER"],
                        num_to_keep=10,
                    )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"], replacements={"DATA_ROOT": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # Sort training data to speed up training and get better results
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # When sorting do not shuffle in dataloader otherwise it is pointless
        hparams["train_dataloader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        # Sort training data to speed up training and get better results
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # When sorting do not shuffle in dataloader otherwise it is pointless
        hparams["train_dataloader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("`sorting` must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration", reverse=True)

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data so it is faster to test
    test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline
    @sb.utils.data_pipeline.takes("mixed_wav", "enroll_wav")
    @sb.utils.data_pipeline.provides("mixed_sig", "enroll_sig")
    def audio_pipeline(mixed_wav, enroll_wav):
        # Mixed signal
        sample_rate = torchaudio.info(mixed_wav).sample_rate
        mixed_sig = sb.dataio.dataio.read_audio(mixed_wav)
        resampled_mixed_sig = torchaudio.transforms.Resample(
            sample_rate, hparams["sample_rate"],
        )(mixed_sig)
        yield resampled_mixed_sig
        # Enrollment signal
        sample_rate = torchaudio.info(enroll_wav).sample_rate
        enroll_sig = sb.dataio.dataio.read_audio(enroll_wav)
        resampled_enroll_sig = torchaudio.transforms.Resample(
            sample_rate, hparams["sample_rate"],
        )(enroll_sig)
        yield resampled_enroll_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_bos", "tokens_eos", "tokens", "target_words",
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        tokens_bos = torch.LongTensor([hparams["blank_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["blank_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        target_words = wrd.split(" ")
        # When `ref_tokens` is an empty string add dummy space
        # to avoid division by 0 when computing WER/CER
        for i, char in enumerate(target_words):
            if len(char) == 0:
                target_words[i] = " "
        yield target_words

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "mixed_sig",
            "enroll_sig",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "target_words",
        ],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation
    from librispeech_mix_prepare import prepare_librispeech_mix  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_librispeech_mix,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
        },
    )

    # Define tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["vocab_size"],
        annotation_train=hparams["train_json"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        annotation_format="json",
    )

    # Create the datasets objects as well as tokenization and encoding
    train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

    # Download the pretrained models
    if hparams["run_pretrainer"]:
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    brain = TSASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Add objects to trainer
    brain.tokenizer = tokenizer

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test on each split separately
    for split in ["test-clean-1mix", "test-clean-2mix", "test-clean-3mix"]:
        # Due to DDP, do the preparation ONLY on the main Python process
        run_on_main(
            prepare_librispeech_mix,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "splits": [split],
            },
        )

        # Create the datasets objects as well as tokenization and encoding
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], f"wer_{split}.txt"
        )
        brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_kwargs"],
        )

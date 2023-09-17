#!/usr/bin/env/python

"""Recipe for training a transducer-based TS-ASR system
(see https://arxiv.org/abs/2209.04175).

A pretrained speaker verification model (kept frozen) is used as a speaker encoder.

To run this recipe, do the following:
> python train_librispeechmix_pretrained.py hparams/LibriSpeechMix/<config>_<speaker-encoder>.yaml

Authors
 * Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/recipes/LibriSpeech/ASR/transducer/train.py

import math
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.linear import Linear
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main
from transformers import AutoModelForAudioXVector


class TSASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        current_epoch = self.hparams.epoch_counter.current

        batch = batch.to(self.device)
        mixed_sigs, mixed_sigs_lens = batch.mixed_sig
        enroll_sigs, enroll_sigs_lens = batch.enroll_sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Extract speaker embedding
        with torch.no_grad():
            self.modules.speaker_encoder.eval()
            speaker_embs = self.modules.speaker_encoder(
                input_values=enroll_sigs,
                attention_mask=length_to_mask(
                    (enroll_sigs_lens * enroll_sigs.shape[-1])
                    .ceil()
                    .clamp(max=enroll_sigs.shape[-1])
                    .int()
                ).long(),  # 0 for masked tokens
                output_attentions=False,
                output_hidden_states=False,
            ).embeddings[:, None, :]
        speaker_embs = self.modules.speaker_proj(speaker_embs)

        # Add speed perturbation if specified
        if self.hparams.augment and stage == sb.Stage.TRAIN:
            if "speed_perturb" in self.modules:
                mixed_sigs = self.modules.speed_perturb(mixed_sigs)

        # Extract features
        feats = self.modules.feature_extractor(mixed_sigs)
        feats = self.modules.normalizer(feats, mixed_sigs_lens, epoch=current_epoch)

        # Add augmentation if specified
        if self.hparams.augment and stage == sb.Stage.TRAIN:
            if "augmentation" in self.modules:
                feats = self.modules.augmentation(feats)

        # Forward encoder/transcriber
        feats = self.modules.frontend(feats)
        enc_out = self.modules.encoder(feats, mixed_sigs_lens, speaker_embs)
        enc_out = self.modules.encoder_proj(enc_out)

        # Forward decoder/predictor
        embs = self.modules.embedding(tokens_bos)
        if self.hparams.decoder_biasing:
            dec_hidden = self.modules.decoder_hidden_proj(speaker_embs)
            dec_hidden = dec_hidden.reshape(
                -1, 2, self.hparams.decoder_num_layers, self.hparams.decoder_neurons
            )
            dec_hidden = (
                dec_hidden[:, 0].movedim(0, 1),
                dec_hidden[:, 1].movedim(0, 1),
            )
        else:
            dec_hidden = None
        dec_out, _ = self.modules.decoder(embs, lengths=tokens_bos_lens, hx=dec_hidden)
        dec_out = self.modules.decoder_proj(dec_out)

        # Forward joiner
        # Add target sequence dimension to the encoder tensor: [B, T, H_enc] => [B, T, 1, H_enc]
        # Add source sequence dimension to the decoder tensor: [B, U, H_dec] => [B, 1, U, H_dec]
        joiner_out = self.modules.joiner(enc_out[..., None, :], dec_out[:, None, ...])

        # Compute transducer logits
        logits = self.modules.transducer_head(joiner_out)

        # Compute outputs
        ctc_logprobs = None
        ce_logprobs = None
        speaker_embs_pred = None
        target_sigs_pred = None
        target_sigs = None
        hyps = None

        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.num_ctc_epochs:
                # Output layer for CTC log-probabilities
                ctc_logits = self.modules.encoder_head(enc_out)
                ctc_logprobs = ctc_logits.log_softmax(dim=-1)

            if current_epoch <= self.hparams.num_ce_epochs:
                # Output layer for CE log-probabilities
                ce_logits = self.modules.decoder_head(dec_out)
                ce_logprobs = ce_logits.log_softmax(dim=-1)

            if current_epoch <= self.hparams.num_speaker_embedding_pred_epochs:
                # Output layer for speaker embedding prediction
                speaker_embs_pred = self.modules.speaker_embedding_pred_proj(enc_out)
                speaker_embs_mask = length_to_mask(
                    (enroll_sigs_lens * speaker_embs_pred.shape[-2])
                    .ceil()
                    .clamp(max=speaker_embs_pred.shape[-2])
                    .int()
                )[
                    ..., None
                ]  # False for masked tokens
                speaker_embs_pred = (speaker_embs_pred * speaker_embs_mask).sum(
                    dim=-2, keepdims=True
                ) / speaker_embs_mask.sum(dim=-2, keepdims=True)

            if "target_sig_pred_proj" in self.modules:
                # Target signal prediction
                target_sigs_pred = self.modules.target_sig_pred_proj(enc_out)
                target_sigs, target_sigs_lens = batch.target_sig
                with torch.no_grad():
                    # Extract features
                    self.modules.feature_extractor.eval()
                    self.modules.normalizer.eval()
                    target_feats = self.modules.feature_extractor(target_sigs)
                    target_feats = self.modules.normalizer(
                        target_feats, target_sigs_lens, epoch=current_epoch
                    )

                    # Forward encoder/transcriber
                    self.modules.frontend.eval()
                    self.modules.encoder.eval()
                    self.modules.encoder_proj.eval()
                    target_feats = self.modules.frontend(target_feats)
                    target_enc_out = self.modules.encoder(
                        target_feats, target_sigs_lens, speaker_embs
                    )
                    target_enc_out = self.modules.encoder_proj(target_enc_out)
                    target_sigs = target_enc_out

        elif stage == sb.Stage.VALID:
            # During validation, run decoding only every valid_search_freq epochs to speed up training
            if current_epoch % self.hparams.valid_search_freq == 0:
                hyps, scores, _, _ = self.hparams.greedy_searcher(enc_out)

        else:
            hyps, scores, _, _ = self.hparams.beam_searcher(enc_out)

        return (
            logits,
            ctc_logprobs,
            ce_logprobs,
            speaker_embs_pred,
            speaker_embs,
            target_sigs_pred,
            target_sigs,
            hyps,
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes the transducer loss + (CTC + CE) given predictions and targets."""
        (
            logits,
            ctc_logprobs,
            ce_logprobs,
            speaker_embs_pred,
            speaker_embs,
            target_sigs_pred,
            target_sigs,
            hyps,
        ) = predictions

        ids = batch.id
        _, mixed_sigs_lens = batch.mixed_sig
        _, enroll_sigs_lens = batch.enroll_sig
        _, target_sigs_lens = batch.target_sig
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.transducer_loss(
            logits, tokens, mixed_sigs_lens, tokens_lens
        )
        if ctc_logprobs is not None:
            loss += self.hparams.ctc_weight * self.hparams.ctc_loss(
                ctc_logprobs, tokens, mixed_sigs_lens, tokens_lens
            )
        if ce_logprobs is not None:
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            loss += self.hparams.ce_weight * self.hparams.ce_loss(
                ce_logprobs, tokens_eos, length=tokens_eos_lens
            )
        if speaker_embs_pred is not None:
            loss += self.hparams.speaker_embedding_pred_weight * self.hparams.mse_loss(
                speaker_embs_pred, speaker_embs, enroll_sigs_lens
            )
        if target_sigs_pred is not None:
            loss += self.hparams.target_sig_pred_weight * self.hparams.mse_loss(
                target_sigs_pred, target_sigs, target_sigs_lens
            )

        if hyps is not None:
            target_words = batch.target_words

            # Decode predicted tokens to words
            predicted_words = self.tokenizer(hyps, task="decode_from_list")

            self.cer_metric.append(ids, predicted_words, target_words)
            self.wer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``, meant for calculating and logging metrics."""
        if self.hparams.enable_scheduler and should_step:
            self.hparams.noam_scheduler(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            if current_epoch % self.hparams.valid_search_freq == 0:
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_scheduler.current_lr
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if current_epoch % self.hparams.valid_search_freq == 0:
                if if_main_process():
                    self.checkpointer.save_and_keep_only(
                        meta={"WER": stage_stats["WER"]},
                        min_keys=["WER"],
                        num_to_keep=self.hparams.keep_checkpoints,
                    )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch}, test_stats=stage_stats,
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
        # Sort training data to speed up training
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["train_remove_if_longer"]},
        )

    elif hparams["sorting"] == "descending":
        # Sort training data to speed up training
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["train_remove_if_longer"]},
        )

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("`sorting` must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline
    @sb.utils.data_pipeline.takes(
        "wavs", "enroll_wav", "delays", "start", "duration", "target_speaker_idx"
    )
    @sb.utils.data_pipeline.provides("mixed_sig", "target_sig", "enroll_sig")
    def audio_pipeline(wavs, enroll_wav, delays, start, duration, target_speaker_idx):
        # Mixed signal
        sigs = []
        for i, (wav, delay) in enumerate(zip(wavs, delays)):
            try:
                sig, sample_rate = torchaudio.load(wav)
            except RuntimeError:
                sig, sample_rate = torchaudio.load(wav.replace(".wav", ".flac"))
            sig = torchaudio.functional.resample(
                sig[0], sample_rate, hparams["sample_rate"],
            )
            if i != target_speaker_idx:
                sig = torchaudio.functional.gain(sig, hparams["gain_nontarget"])
            frame_delay = math.ceil(delay * hparams["sample_rate"])
            sig = torch.nn.functional.pad(sig, [frame_delay, 0])
            sigs.append(sig)
        max_length = max(len(x) for x in sigs)
        sigs = [torch.nn.functional.pad(x, [0, max_length - len(x)]) for x in sigs]
        mixed_sig = sigs[0].clone()
        for sig in sigs[1:]:
            mixed_sig += sig
        frame_start = math.ceil(start * hparams["sample_rate"])
        frame_duration = math.ceil(duration * hparams["sample_rate"])
        mixed_sig = mixed_sig[frame_start : frame_start + frame_duration]
        yield mixed_sig

        # Target signal
        yield sigs[target_speaker_idx][frame_start : frame_start + frame_duration]

        # Enrollment signal
        try:
            enroll_sig, sample_rate = torchaudio.load(enroll_wav)
        except RuntimeError:
            enroll_sig, sample_rate = torchaudio.load(
                enroll_wav.replace(".wav", ".flac")
            )
        enroll_sig = torchaudio.functional.resample(
            enroll_sig[0], sample_rate, hparams["sample_rate"],
        )
        # Trim enrollment signal if too long
        enroll_sig = enroll_sig[
            : math.ceil(hparams["max_enroll_length"] * hparams["sample_rate"])
        ]
        yield enroll_sig

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
            "target_sig",
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
    from librispeechmix_prepare import prepare_librispeechmix  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_librispeechmix,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "num_targets": hparams["num_targets"],
            "num_enrolls": hparams["num_enrolls"],
            "trim_nontarget": hparams["trim_nontarget"],
            "suppress_delay": hparams["suppress_delay"],
            "overlap_ratio": hparams["overlap_ratio"],
        },
    )

    # Define tokenizer
    # NOTE: the token distribution of the train set might differ from that of the dev/test
    # set (in this case you should fit the tokenizer on both train, dev, and test)
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["vocab_size"],
        annotation_train=hparams["train_json"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        unk_id=hparams["blank_index"],
        annotation_format="json",
    )

    # Create the datasets objects as well as tokenization and encoding
    train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

    # Pretrain the specified modules
    run_on_main(hparams["pretrainer"].collect_files)
    run_on_main(hparams["pretrainer"].load_collected)

    # Download the pretrained speaker encoder
    speaker_encoder = AutoModelForAudioXVector.from_pretrained(
        hparams["speaker_encoder_path"]
    )
    hparams["modules"]["speaker_encoder"] = speaker_encoder

    # Log number of parameters in the speaker encoder
    sb.core.logger.info(
        f"{round(sum([x.numel() for x in speaker_encoder.parameters()]) / 1e6)}M parameters in frozen speaker encoder"
    )

    # Add module for decoder biasing
    if hparams["decoder_biasing"]:
        decoder_hidden_proj = Linear(
            input_size=hparams["d_model"],
            n_neurons=2 * hparams["decoder_neurons"] * hparams["decoder_num_layers"],
        )
        hparams["modules"]["decoder_hidden_proj"] = decoder_hidden_proj
        hparams["model"].append(decoder_hidden_proj)
        hparams["checkpointer"].add_recoverable(
            "decoder_hidden_proj", decoder_hidden_proj
        )

    # Add module for speaker embedding prediction
    if hparams["num_speaker_embedding_pred_epochs"] > 0:
        speaker_embedding_pred_proj = Sequential(
            Linear(
                input_size=hparams["joint_dim"], n_neurons=hparams["auxiliary_neurons"]
            ),
            torch.nn.ReLU(),
            Linear(
                input_size=hparams["auxiliary_neurons"], n_neurons=hparams["d_model"]
            ),
        )
        hparams["modules"]["speaker_embedding_pred_proj"] = speaker_embedding_pred_proj
        hparams["model"].append(speaker_embedding_pred_proj)
        hparams["checkpointer"].add_recoverable(
            "speaker_embedding_pred_proj", speaker_embedding_pred_proj
        )

    # Add module for target signal prediction
    if hparams["target_sig_pred_weight"] > 0:
        target_sig_pred_proj = Sequential(
            Linear(
                input_size=hparams["joint_dim"], n_neurons=hparams["auxiliary_neurons"]
            ),
            torch.nn.ReLU(),
            Linear(
                input_size=hparams["auxiliary_neurons"], n_neurons=hparams["joint_dim"]
            ),
        )
        hparams["modules"]["target_sig_pred_proj"] = target_sig_pred_proj
        hparams["model"].append(target_sig_pred_proj)
        hparams["checkpointer"].add_recoverable(
            "target_sig_pred_proj", target_sig_pred_proj
        )

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

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {"num_workers": hparams["dataloader_workers"]}
    if hparams["dynamic_batching"]:
        hparams["train_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams["sorting"],
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams["train_batch_size"]

    hparams["valid_dataloader_kwargs"] = {"num_workers": hparams["dataloader_workers"]}
    if hparams["dynamic_batching"]:
        hparams["valid_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams["valid_batch_size"]

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test on each split separately
    for split in hparams["test_splits"]:
        # Due to DDP, do the preparation ONLY on the main Python process
        run_on_main(
            prepare_librispeechmix,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "splits": [split],
                "num_targets": hparams["num_targets"],
                "num_enrolls": hparams["num_enrolls"],
                "trim_nontarget": hparams["trim_nontarget"],
                "suppress_delay": hparams["suppress_delay"],
                "overlap_ratio": hparams["overlap_ratio"],
            },
        )

        # Create the datasets objects as well as tokenization and encoding
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        # Dynamic batching
        hparams["test_dataloader_kwargs"] = {
            "num_workers": hparams["dataloader_workers"]
        }
        if hparams["dynamic_batching"]:
            hparams["test_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
                test_data,
                hparams["test_max_batch_length"],
                num_buckets=hparams["num_buckets"],
                length_func=lambda x: x["duration"],
                shuffle=False,
                batch_ordering="descending",
                max_batch_ex=hparams["max_batch_size"],
            )
        else:
            hparams["test_dataloader_kwargs"]["batch_size"] = hparams["test_batch_size"]

        brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], f"wer_{split}.txt"
        )

        brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_kwargs"],
        )

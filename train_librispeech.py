#!/usr/bin/env/python

"""Recipe for training a transducer-based ASR system.

To run this recipe, do the following:
> python train_librispeech.py hparams/LibriSpeech/<config>.yaml

Authors
 * Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/recipes/LibriSpeech/ASR/transducer/train.py

import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import ddp_barrier, if_main_process, run_on_main


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wavs_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add speed perturbation if specified
        if self.hparams.augment and stage == sb.Stage.TRAIN:
            if "speed_perturb" in self.modules:
                wavs = self.modules.speed_perturb(wavs)

        # Extract features
        feats = self.modules.feature_extractor(wavs)

        # Add augmentation if specified
        if self.hparams.augment and stage == sb.Stage.TRAIN:
            if "augmentation" in self.modules:
                feats = self.modules.augmentation(feats)

        # Forward encoder/transcriber
        feats = self.modules.frontend(feats)
        enc_out = self.modules.encoder(feats, wavs_lens)

        # Forward decoder/predictor
        embs = self.modules.embedding(tokens_bos)
        dec_out, _ = self.modules.decoder(embs, lengths=tokens_bos_lens)

        # Forward joiner
        # Add label dimension to the encoder tensor: [B, T, H_enc] => [B, T, 1, H_enc]
        # Add time dimension to the decoder tensor: [B, U, H_dec] => [B, 1, U, H_dec]
        join_out = self.modules.joiner(enc_out[..., None, :], dec_out[:, None, ...])

        # Compute transducer log-probabilities
        logits = self.modules.transducer_head(join_out)

        # Compute outputs
        ctc_logprobs = None
        ce_logprobs = None
        hyps = None

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.num_ctc_epochs:
                # Output layer for CTC log-probabilities
                ctc_logits = self.modules.encoder_head(enc_out)
                ctc_logprobs = ctc_logits.log_softmax(dim=-1)

            if current_epoch <= self.hparams.num_ce_epochs:
                # Output layer for CE log-probabilities
                ce_logits = self.modules.decoder_head(dec_out)
                ce_logprobs = ce_logits.log_softmax(dim=-1)

        elif stage == sb.Stage.VALID:
            # During validation, run decoding only every valid_search_freq epochs to speed up training
            if current_epoch % self.hparams.valid_search_freq == 0:
                hyps, scores, _, _ = self.hparams.greedy_searcher(enc_out)

        else:
            hyps, scores, _, _ = self.hparams.beam_searcher(enc_out)

        return logits, ctc_logprobs, ce_logprobs, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the transducer loss + (CTC + CE) given predictions and targets."""
        logits, ctc_logprobs, ce_logprobs, hyps = predictions

        ids = batch.id
        _, wavs_lens = batch.sig
        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = self.hparams.transducer_loss(logits, tokens, wavs_lens, tokens_lens)
        if ctc_logprobs is not None:
            loss += self.hparams.ctc_weight * self.hparams.ctc_loss(
                ctc_logprobs, tokens, wavs_lens, tokens_lens
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
        """Fit one batch, override to do multiple updates."""
        should_step = (self.valid_step + 1) % self.grad_accumulation_factor == 0

        # should_step=True => synchronize gradient between DDP processes
        with self.no_sync(not should_step):
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16 if self.bfloat16_mix_prec else torch.float16,
                enabled=self.auto_mix_prec,
            ):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.check_gradients(loss)

            if self.auto_mix_prec:
                self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            else:
                (loss / self.grad_accumulation_factor).backward()

            if should_step:
                if self.auto_mix_prec:
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.max_grad_norm
                    )
                if self.auto_mix_prec:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.zero_grad(set_to_none=True)
                self.optimizer_step += 1
            self.valid_step += 1

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def check_gradients(self, loss):
        if loss.isfinite():
            return True

        self.nonfinite_count += 1

        # Print helpful debug info
        sb.core.logger.warning(f"Loss is {loss}.")
        for p in self.modules.parameters():
            if not torch.isfinite(p).all():
                sb.core.logger.warning("Parameter is not finite: " + str(p))

        # Check if patience is exhausted
        if self.nonfinite_count > self.nonfinite_patience:
            raise ValueError(
                "Loss is not finite and patience is exhausted. "
                "To debug, wrap `fit()` with "
                "autograd's `detect_anomaly()`, e.g.\n\nwith "
                "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )
        sb.core.logger.warning("Patience not yet exhausted, ignoring this batch.")
        return False

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

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"DATA_ROOT": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # Sort training data to speed up training and get better results
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["train_remove_if_longer"]},
        )

    elif hparams["sorting"] == "descending":
        # Sort training data to speed up training and get better results
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["train_remove_if_longer"]},
        )

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("`sorting` must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    # Sort the test data so it is faster to test
    test_data = {}
    for split, test_csv in zip(hparams["test_splits"], hparams["test_csv"]):
        test_data[split] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=test_csv, replacements={"DATA_ROOT": data_folder},
        )
        test_data[split].filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["test_remove_if_longer"]},
        )

    datasets = [train_data, valid_data, *test_data.values()]

    # 2. Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # Signal
        sample_rate = torchaudio.info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        resampled_sig = torchaudio.transforms.Resample(
            sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled_sig

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
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens", "target_words",],
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
    from librispeech_prepare import prepare_librispeech  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["save_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Define tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["vocab_size"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        unk_id=hparams["blank_index"],
        annotation_format="csv",
    )

    # Create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    brain = ASR(
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
    for split in test_data:
        hparams["test_dataloader_kwargs"] = {
            "num_workers": hparams["dataloader_workers"]
        }
        if hparams["dynamic_batching"]:
            hparams["test_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
                test_data[split],
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
            test_data[split],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_kwargs"],
        )

"""Datasets.

Authors
* Luca Della Libera 2023
"""

import os
import itertools
from contextlib import nullcontext

import sentencepiece
import torch
import torchaudio
import torchmetrics
from torch import distributed as dist
from tqdm import tqdm
from torchaudio import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForAudioXVector

from collators import PaddedBatch
from datasets import LibriSpeechMix, TokenizedDataset
from models import conformer_rnnt_model


# Configuration
SEED = 0
LIBRISPEECH_PATH = "../datasets/LibriSpeech"
TRAIN_JSONL = "../datasets/LibriSpeechMix/list/train-clean-2mix.jsonl"
VALID_JSONL = "../datasets/LibriSpeechMix/list/dev-clean-2mix.jsonl"
TEST_JSONL = "../datasets/LibriSpeechMix/list/test-clean-2mix.jsonl"

DEBUG = False

OUTPUT_DIR = os.path.join("LibriSpeechMix", "conformer-t")
SAVE_DIR = os.path.join(OUTPUT_DIR, "save")

MAX_ENROLLS = 1
SUPPRESS_DELAY = True
GAIN_NONTARGET = -40
USE_SPEAKER_EMBS = True

SAMPLE_RATE = 16000
MAX_ENROLL_LENGTH = 10
VOCAB_SIZE = 1000
D_MODEL = 256
ENCODER_FFN_DIM = 2048
ENCODER_NUM_LAYERS = 12
TIME_REDUCTION_STRIDE = 2
SYMBOL_EMBEDDING_DIM = 256
BLANK_IDX = 0
VALID_SEARCH_FREQ = 10
BEAM_WIDTH = 4

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

LR = 1e-2
NUM_EPOCHS = 100
GRAD_ACCUMULATION_FACTOR = 4
MAX_GRAD_L2_NORM = 30
USE_AMP = True
###############


if __name__ == "__main__":
    is_distributed = "LOCAL_RANK" in os.environ

    # Handle DDP
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    # Set seed
    torch.cuda.manual_seed_all(SEED)

    # Set device
    if is_distributed and dist.dist_backend != "nccl":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(rank)

    # Load data
    train_dataset = LibriSpeechMix(
        LIBRISPEECH_PATH,
        TRAIN_JSONL,
        max_enrolls=MAX_ENROLLS,
        gain_nontarget=GAIN_NONTARGET,
        suppress_delay=SUPPRESS_DELAY,
    )
    valid_dataset = LibriSpeechMix(
        LIBRISPEECH_PATH,
        VALID_JSONL,
        max_enrolls=MAX_ENROLLS,
        gain_nontarget=GAIN_NONTARGET,
        suppress_delay=SUPPRESS_DELAY,
    )
    test_dataset = LibriSpeechMix(
        LIBRISPEECH_PATH,
        TEST_JSONL,
        max_enrolls=MAX_ENROLLS,
        gain_nontarget=GAIN_NONTARGET,
        suppress_delay=SUPPRESS_DELAY,
    )

    # Train tokenizer
    texts_file = os.path.join(SAVE_DIR, "train.txt")
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer", "sp")
    if rank == 0:
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(texts_file, "w") as f:
            f.write("\n".join(train_dataset.texts[: 5 if DEBUG else None]))
        sentencepiece.SentencePieceTrainer.train(
            input=texts_file,
            model_prefix=tokenizer_path,
            vocab_size=VOCAB_SIZE,
            character_coverage=1.0,
            model_type="unigram",
            unk_id=BLANK_IDX,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
    if is_distributed:
        dist.barrier()
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=f"{tokenizer_path}.model"
    )

    # Tokenize
    train_dataset = TokenizedDataset(train_dataset, tokenizer.encode)
    valid_dataset = TokenizedDataset(valid_dataset, tokenizer.encode)
    test_dataset = TokenizedDataset(test_dataset, tokenizer.encode)
    if DEBUG:
        train_dataset = Subset(
            TokenizedDataset(train_dataset, tokenizer.encode), range(5)
        )
        valid_dataset = Subset(
            TokenizedDataset(valid_dataset, tokenizer.encode), range(5)
        )
        test_dataset = Subset(
            TokenizedDataset(test_dataset, tokenizer.encode), range(5)
        )

    # Build dataloaders
    collator = PaddedBatch(SAMPLE_RATE, MAX_ENROLL_LENGTH, BLANK_IDX)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True if not is_distributed else False,
        num_workers=6,
        sampler=DistributedSampler(train_dataset) if is_distributed else None,
        collate_fn=collator,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=6,
        sampler=DistributedSampler(valid_dataset) if is_distributed else None,
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        num_workers=6,
        sampler=DistributedSampler(test_dataset) if is_distributed else None,
        collate_fn=collator,
    )

    # Define feature extractor
    feature_extractor = torch.nn.Sequential(
        transforms.Resample(SAMPLE_RATE),
        transforms.MelSpectrogram(SAMPLE_RATE, n_mels=80),
    ).to(device)

    # Define augmentation
    augmentation = torch.nn.Sequential(
        transforms.TimeMasking(time_mask_param=80),
        transforms.FrequencyMasking(freq_mask_param=80),
    ).to(device)

    # Define model
    model = conformer_rnnt_model(
        input_dim=80,
        encoding_dim=D_MODEL,
        transformer_ffn_dim=ENCODER_FFN_DIM,
        transformer_num_layers=ENCODER_NUM_LAYERS,
        transformer_depthwise_conv_kernel_size=31,
        time_reduction_stride=TIME_REDUCTION_STRIDE,
        symbol_embedding_dim=SYMBOL_EMBEDDING_DIM,
        num_symbols=tokenizer.vocab_size(),
    ).to(device)

    # Define searcher
    searcher = torchaudio.models.RNNTBeamSearch(
        model, blank=BLANK_IDX, step_max_tokens=1
    )

    # Download the pretrained speaker encoder
    speaker_encoder = AutoModelForAudioXVector.from_pretrained(
        "microsoft/wavlm-base-sv"
    ).to(device)
    speaker_proj = torch.nn.Linear(speaker_encoder.config.xvector_output_dim, 80).to(
        device
    )

    # Define optimizer
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *speaker_proj.parameters()], lr=LR
    )

    # Define scaler
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Initialize counters
    start_epoch = 0
    optimizer_steps = 0

    # Open log file
    if rank == 0:
        train_log = open(os.path.join(OUTPUT_DIR, "train_log.txt"), "a")

    checkpoint_paths = sorted(
        [x for x in os.listdir(SAVE_DIR) if x.startswith("epoch")]
    )
    if checkpoint_paths:
        checkpoint_path = os.path.join(SAVE_DIR, checkpoint_paths[-1])
        checkpoint = torch.load(checkpoint_path)
        torch.random.set_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer_steps = checkpoint["optimizer_steps"] + 1
        model.load_state_dict(checkpoint["model"])
        speaker_proj.load_state_dict(checkpoint["speaker_proj"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if rank == 0:
            print(f"Loaded checkpoint {checkpoint_path}")
            train_log.write(f"Saved checkpoint {checkpoint_path}\n")

    # Handle DDP
    if is_distributed:
        wrapped_model = DDP(model)
        wrapped_speaker_proj = DDP(speaker_proj, find_unused_parameters=True)
    else:
        wrapped_model = model
        wrapped_speaker_proj = speaker_proj

    if rank == 0:
        num_parameters = round(
            sum([x.numel() for x in [*model.parameters(), *speaker_proj.parameters()]])
            / 1e6,
            1,
        )
        print(f"Parameters: {num_parameters}M")
        train_log.write(f"Parameters: {num_parameters}M\n")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_avg_loss = 0.0
        wrapped_model.train()
        wrapped_speaker_proj.train()
        if is_distributed:
            dist.barrier()
        with tqdm(train_dataloader) as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                should_step = (batch_idx + 1) % GRAD_ACCUMULATION_FACTOR == 0
                mixed_sig, mixed_sig_length = [x.to(device) for x in data["mixed_sig"]]
                enroll_sig, enroll_sig_length = [
                    x.to(device) for x in data["enroll_sig"]
                ]
                tokens, tokens_length = [x.to(device) for x in data["tokens"]]
                bos_tokens, bos_tokens_length = [
                    x.to(device) for x in data["bos_tokens"]
                ]
                with wrapped_model.no_sync() if hasattr(
                    wrapped_model, "no_sync"
                ) and not should_step else nullcontext():
                    with wrapped_speaker_proj.no_sync() if hasattr(
                        wrapped_speaker_proj, "no_sync"
                    ) and not should_step else nullcontext():
                        with torch.autocast(
                            device.type,
                            dtype=torch.bfloat16
                            if device.type == "cpu"
                            else torch.float16,
                            enabled=USE_AMP,
                        ):
                            # Extract features
                            feats = feature_extractor(mixed_sig).movedim(-1, -2)
                            feats_lengths = (
                                (
                                    (mixed_sig_length / mixed_sig.shape[-1])
                                    * feats.shape[-2]
                                )
                                .round()
                                .int()
                            )

                            # Augmentation
                            feats = augmentation(feats)

                            # Extract speaker embedding
                            with torch.autocast(device.type, enabled=False):
                                with torch.no_grad():
                                    speaker_encoder.eval()
                                    attention_mask = (
                                        torch.arange(
                                            enroll_sig_length.max(),
                                            device=enroll_sig_length.device,
                                        )[None, :]
                                        < enroll_sig_length[:, None]
                                    )
                                    speaker_embs = speaker_encoder(
                                        input_values=enroll_sig,
                                        attention_mask=attention_mask.long(),  # 0 for masked tokens
                                        output_attentions=False,
                                        output_hidden_states=False,
                                    ).embeddings[:, None, :]

                            # Forward pass
                            speaker_embs = wrapped_speaker_proj(speaker_embs)
                            if USE_SPEAKER_EMBS:
                                feats += speaker_embs
                            logits, logits_lengths, target_lengths, _ = wrapped_model(
                                feats, feats_lengths, bos_tokens, bos_tokens_length
                            )
                            with torch.autocast(device.type, enabled=False):
                                loss = torchaudio.functional.rnnt_loss(
                                    logits.float(),
                                    tokens,
                                    logits_lengths,
                                    tokens_length,
                                    blank=BLANK_IDX,
                                )

                        if should_step:
                            scaler.scale(loss / GRAD_ACCUMULATION_FACTOR).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                wrapped_model.parameters(), MAX_GRAD_L2_NORM
                            )
                            torch.nn.utils.clip_grad_norm_(
                                wrapped_speaker_proj.parameters(), MAX_GRAD_L2_NORM
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1
                train_avg_loss += loss.item() / len(train_dataloader)
                if rank == 0:
                    progress_bar.set_postfix(
                        epoch=epoch,
                        optimizer_steps=optimizer_steps,
                        train_loss=train_avg_loss,
                    )

        # Validation
        if is_distributed:
            dist.barrier()
        should_search = (epoch + 1) % VALID_SEARCH_FREQ == 0
        valid_avg_loss = 0.0
        hyps, refs = [], []
        model.eval()
        speaker_proj.eval()
        with tqdm(valid_dataloader) as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                mixed_sig, mixed_sig_length = [x.to(device) for x in data["mixed_sig"]]
                enroll_sig, enroll_sig_length = [
                    x.to(device) for x in data["enroll_sig"]
                ]
                tokens, tokens_length = [x.to(device) for x in data["tokens"]]
                bos_tokens, bos_tokens_length = [
                    x.to(device) for x in data["bos_tokens"]
                ]

                # Extract features
                feats = feature_extractor(mixed_sig).movedim(-1, -2)
                feats_lengths = (
                    ((mixed_sig_length / mixed_sig.shape[-1]) * feats.shape[-2])
                    .round()
                    .int()
                )

                # Extract speaker embedding
                with torch.no_grad():
                    speaker_encoder.eval()
                    attention_mask = (
                        torch.arange(
                            enroll_sig_length.max(), device=enroll_sig_length.device
                        )[None, :]
                        < enroll_sig_length[:, None]
                    )
                    speaker_embs = speaker_encoder(
                        input_values=enroll_sig,
                        attention_mask=attention_mask.long(),  # 0 for masked tokens
                        output_attentions=False,
                        output_hidden_states=False,
                    ).embeddings[:, None, :]

                # Forward pass
                with torch.no_grad():
                    speaker_embs = speaker_proj(speaker_embs)
                    if USE_SPEAKER_EMBS:
                        feats += speaker_embs
                    logits, logits_lengths, target_lengths, _ = model(
                        feats, feats_lengths, bos_tokens, bos_tokens_length
                    )
                    loss = torchaudio.functional.rnnt_loss(
                        logits, tokens, logits_lengths, tokens_length, blank=BLANK_IDX
                    )

                # Search
                if should_search:
                    for feat, feat_length, ref, ref_length in zip(
                        feats, feats_lengths, tokens, tokens_length
                    ):
                        with torch.no_grad():
                            hyp, _, _, _ = searcher(
                                feat[:feat_length][None],
                                feat_length[None],
                                beam_width=1,
                            )[0]
                            hyp = [x for x in hyp if x != BLANK_IDX]
                        hyps.append(tokenizer.decode(hyp))
                        refs.append(tokenizer.decode(ref[:ref_length].tolist()))

                valid_avg_loss += loss.item() / len(valid_dataloader)

                if not should_search or batch_idx < len(progress_bar) - 1:
                    if rank == 0:
                        progress_bar.set_postfix(epoch=epoch, valid_loss=valid_avg_loss)
                else:
                    if is_distributed:
                        dist.barrier()
                        all_hyps = [None] * int(os.environ["WORLD_SIZE"])
                        all_refs = [None] * int(os.environ["WORLD_SIZE"])
                        dist.all_gather_object(all_hyps, hyps)
                        dist.all_gather_object(all_refs, refs)
                        # print(all_refs)
                        hyps = list(itertools.chain(*all_hyps))
                        refs = list(itertools.chain(*all_refs))
                    # print(refs)
                    # print(hyps)
                    print(rank, len(hyps), len(refs))
                    if rank == 0:
                        cer_computer = torchmetrics.text.CharErrorRate()
                        wer_computer = torchmetrics.text.WordErrorRate()
                        cer = cer_computer(hyps, refs).item() * 100
                        wer = wer_computer(hyps, refs).item() * 100
                        progress_bar.set_postfix(
                            epoch=epoch,
                            valid_loss=valid_avg_loss,
                            CER=f"{cer}%",
                            WER=f"{wer}%",
                        )
                        train_log.write(
                            f"epoch={epoch}, optimizer_steps={optimizer_steps}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, CER={cer}%, WER={wer}%\n"
                        )

                        checkpoint = {}
                        checkpoint["rng_state"] = torch.random.get_rng_state()
                        checkpoint["epoch"] = epoch
                        checkpoint["optimizer_steps"] = optimizer_steps
                        checkpoint["model"] = model.state_dict()
                        checkpoint["speaker_proj"] = speaker_proj.state_dict()
                        checkpoint["optimizer"] = optimizer.state_dict()
                        checkpoint["scaler"] = scaler.state_dict()
                        os.makedirs(SAVE_DIR, exist_ok=True)
                        checkpoint_path = os.path.join(
                            SAVE_DIR, f"epoch={str(epoch).zfill(3)}_wer={wer}.pt"
                        )
                        torch.save(checkpoint, checkpoint_path)
                        checkpoint_paths = sorted(
                            [x for x in os.listdir(SAVE_DIR) if x.startswith("epoch")]
                        )
                        if len(checkpoint_paths) > 1:
                            for i in range(len(checkpoint_paths) - 1):
                                os.remove(os.path.join(SAVE_DIR, checkpoint_paths[i]))
                        print(f"Saved checkpoint {checkpoint_path}")
                        train_log.write(f"Saved checkpoint {checkpoint_path}\n")

    # Test
    if is_distributed:
        dist.barrier()
    test_avg_loss = 0.0
    hyps, refs = [], []
    model.eval()
    speaker_proj.eval()
    with tqdm(test_dataloader) as progress_bar:
        for batch_idx, data in enumerate(progress_bar):
            mixed_sig, mixed_sig_length = [x.to(device) for x in data["mixed_sig"]]
            enroll_sig, enroll_sig_length = [x.to(device) for x in data["enroll_sig"]]
            tokens, tokens_length = [x.to(device) for x in data["tokens"]]
            bos_tokens, bos_tokens_length = [x.to(device) for x in data["bos_tokens"]]

            # Extract features
            feats = feature_extractor(mixed_sig).movedim(-1, -2)
            feats_lengths = (
                ((mixed_sig_length / mixed_sig.shape[-1]) * feats.shape[-2])
                .round()
                .int()
            )

            # Extract speaker embedding
            with torch.no_grad():
                speaker_encoder.eval()
                attention_mask = (
                    torch.arange(
                        enroll_sig_length.max(), device=enroll_sig_length.device
                    )[None, :]
                    < enroll_sig_length[:, None]
                )
                speaker_embs = speaker_encoder(
                    input_values=enroll_sig,
                    attention_mask=attention_mask.long(),  # 0 for masked tokens
                    output_attentions=False,
                    output_hidden_states=False,
                ).embeddings[:, None, :]

            # Forward pass
            with torch.no_grad():
                speaker_embs = speaker_proj(speaker_embs)
                if USE_SPEAKER_EMBS:
                    feats += speaker_embs
                logits, logits_lengths, target_lengths, _ = model(
                    feats, feats_lengths, bos_tokens, bos_tokens_length
                )
                loss = torchaudio.functional.rnnt_loss(
                    logits, tokens, logits_lengths, tokens_length, blank=BLANK_IDX
                )

            # Search
            for feat, feat_length, ref, ref_length in zip(
                feats, feats_lengths, tokens, tokens_length
            ):
                with torch.no_grad():
                    hyp, _, _, _ = searcher(
                        feat[:feat_length][None], feat_length[None], beam_width=1
                    )[0]
                    hyp = [x for x in hyp if x != BLANK_IDX]
                hyps.append(tokenizer.decode(hyp))
                refs.append(tokenizer.decode(ref[:ref_length].tolist()))

            test_avg_loss += loss.item() / len(test_dataloader)

            if batch_idx < len(progress_bar) - 1:
                if rank == 0:
                    progress_bar.set_postfix(test_loss=test_avg_loss)
            else:
                if is_distributed:
                    dist.barrier()
                    all_hyps = [None] * int(os.environ["WORLD_SIZE"])
                    all_refs = [None] * int(os.environ["WORLD_SIZE"])
                    dist.all_gather_object(all_hyps, hyps)
                    dist.all_gather_object(all_refs, refs)
                    hyps = list(itertools.chain(*all_hyps))
                    refs = list(itertools.chain(*all_refs))
                if rank == 0:
                    cer_computer = torchmetrics.text.CharErrorRate()
                    wer_computer = torchmetrics.text.WordErrorRate()
                    cer = cer_computer(hyps, refs).item() * 100
                    wer = wer_computer(hyps, refs).item() * 100
                    progress_bar.set_postfix(
                        test_loss=test_avg_loss, CER=f"{cer}%", WER=f"{wer}%"
                    )
                    train_log.write(
                        f"test_loss={test_avg_loss}, CER={cer}%, WER={wer}%\n"
                    )
                    with open(os.path.join(OUTPUT_DIR, "wer_test.txt"), "w") as f:
                        for hyp, ref in zip(hyps, refs):
                            f.write(ref + "\n" + hyp + "\n\n")
                if is_distributed:
                    dist.barrier()

    if rank == 0:
        train_log.close()
    if is_distributed:
        dist.destroy_process_group()

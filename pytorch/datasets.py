"""Datasets.

Authors
* Luca Della Libera 2023
"""

import json
import math
import os
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset


__all__ = ["LibriSpeechMix", "TokenizedDataset"]


class LibriSpeechMix(Dataset):
    def __init__(
        self,
        librispeech_path,
        annotation_jsonl,
        max_enrolls: "Optional[int]" = None,
        gain_nontarget=0.0,
        suppress_delay=False,
    ):
        self.librispeech_path = librispeech_path
        self.annotation_jsonl = annotation_jsonl
        self.max_enrolls = max_enrolls
        self.gain_nontarget = gain_nontarget
        self.suppress_delay = suppress_delay

        # Read annotation JSONL
        self._texts = []
        self._data = []
        with open(self.annotation_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                delays = entry["delays"]
                wavs = [os.path.join(self.librispeech_path, x) for x in entry["wavs"]]
                texts = entry["texts"]
                enrolls = [
                    entry["speaker_profile"][idx][:max_enrolls]
                    for idx in entry["speaker_profile_index"]
                ]
                for target_speaker_idx, text in enumerate(texts):
                    for enroll in enrolls[target_speaker_idx]:
                        self._data.append(
                            {
                                "delays": delays,
                                "wavs": wavs,
                                "enroll": os.path.join(self.librispeech_path, enroll),
                                "text": text,
                                "target_speaker_idx": target_speaker_idx,
                            }
                        )
                self._texts.append(text)

    @property
    def texts(self):
        return self._texts

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]

        # Dynamic mixing with gain
        sigs = []
        for wav in datum["wavs"]:
            try:
                sig, sample_rate = torchaudio.load(wav)
            except RuntimeError:
                sig, sample_rate = torchaudio.load(wav.replace(".wav", ".flac"))
            sig = sig[0]
            sigs.append(sig)

        if self.suppress_delay:
            frame_delays = [0 for _ in datum["delays"]]
        else:
            frame_delays = [math.ceil(d * sample_rate) for d in datum["delays"]]
        max_length = max([len(x) + d for x, d in zip(sigs, frame_delays)])
        mixed_sig = torch.zeros(max_length)
        for i, (sig, frame_delay) in enumerate(zip(sigs, frame_delays)):
            if i != datum["target_speaker_idx"]:
                sig = torchaudio.functional.gain(sig, self.gain_nontarget)
            sig = torch.nn.functional.pad(sig, [frame_delay, 0])
            sig = torch.nn.functional.pad(sig, [0, max_length - len(sig)])
            mixed_sig += sig

        # Enrollment signal
        try:
            enroll_sig, _ = torchaudio.load(datum["enroll"])
        except RuntimeError:
            enroll_sig, _ = torchaudio.load(datum["enroll"].replace(".wav", ".flac"))
        enroll_sig = enroll_sig[0]

        return {"mixed_sig": mixed_sig, "enroll_sig": enroll_sig, "text": datum["text"]}


class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        tokens = self.tokenizer(datum["text"])
        entry = {k: v for k, v in datum.items()}
        entry.update({"tokens": tokens})
        return entry

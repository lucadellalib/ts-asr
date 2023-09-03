"""Batch collators.

Authors
* Luca Della Libera 2023
"""

import torch
from torch.nn import functional as F


__all__ = ["PaddedBatch"]


class PaddedBatch:
    def __init__(self, sample_rate=16000, max_enroll_length=None, blank_idx=0):
        self.sample_rate = sample_rate
        self.max_enroll_length = max_enroll_length
        self.blank_idx = blank_idx

    def __call__(self, batch):
        mixed_sigs = []
        enroll_sigs = []
        tokenss = []
        bos_tokenss = []
        for datum in batch:
            mixed_sig, enroll_sig, tokens = (
                datum["mixed_sig"],
                datum["enroll_sig"],
                datum["tokens"],
            )
            bos_tokens = torch.IntTensor([self.blank_idx] + tokens)
            tokens = torch.IntTensor(tokens)
            mixed_sigs.append(mixed_sig)
            enroll_sigs.append(enroll_sig[: self.max_enroll_length * self.sample_rate])
            tokenss.append(tokens)
            bos_tokenss.append(bos_tokens)

        # Mixed signal
        lengths = torch.IntTensor([x.shape[0] for x in mixed_sigs])
        max_length = max(lengths)
        mixed_sigs = (
            [F.pad(x, [0, max_length - x.shape[0]]) for x in mixed_sigs],
            lengths,
        )

        # Enroll signal
        lengths = torch.IntTensor([x.shape[0] for x in enroll_sigs])
        max_length = max(lengths)
        enroll_sigs = (
            [F.pad(x, [0, max_length - x.shape[0]]) for x in enroll_sigs],
            lengths,
        )

        # Tokens
        lengths = torch.IntTensor([x.shape[0] for x in tokenss])
        max_length = max(lengths)
        tokenss = (
            [
                F.pad(x, [0, max_length - x.shape[0]], value=self.blank_idx)
                for x in tokenss
            ],
            lengths,
        )

        # BOS tokens
        lengths = torch.IntTensor([x.shape[0] for x in bos_tokenss])
        max_length = max(lengths)
        bos_tokenss = (
            [
                F.pad(x, [0, max_length - x.shape[0]], value=self.blank_idx)
                for x in bos_tokenss
            ],
            lengths,
        )

        return {
            "mixed_sig": (torch.stack(mixed_sigs[0]), mixed_sigs[1]),
            "enroll_sig": (torch.stack(enroll_sigs[0]), enroll_sigs[1]),
            "tokens": (torch.stack(tokenss[0]).int(), tokenss[1]),
            "bos_tokens": (torch.stack(bos_tokenss[0]).int(), bos_tokenss[1]),
        }

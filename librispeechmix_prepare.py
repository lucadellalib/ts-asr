"""LibriSpeechMix data preparation.

Authors
 * Luca Della Libera 2023
"""

import copy
import json
import logging
import os
from collections import defaultdict
from typing import List, Optional, Sequence, Union


__all__ = ["prepare_librispeechmix"]


# Logging configuration
logging.basicConfig(
    level=logging.WARNING,  # format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SPLITS = (
    "train-1mix",
    "train-2mix",
    "train-3mix",
    "dev-clean-1mix",
    "dev-clean-2mix",
    "dev-clean-3mix",
    "test-clean-1mix",
    "test-clean-2mix",
    "test-clean-3mix",
)


def prepare_librispeechmix(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = _DEFAULT_SPLITS,
    num_targets: "Optional[Union[int, List[int]], str]" = None,
    num_enrolls: "Optional[int]" = None,
    trim_nontarget: "Optional[float]" = None,
    suppress_delay: "Optional[bool]" = None,
    overlap_ratio: "Optional[float]" = None,
) -> "None":
    """Prepare data manifest JSON files for LibriSpeechMix dataset
    (see https://github.com/NaoyukiKanda/LibriSpeechMix).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder (i.e. a folder containing the standard
        LibriSpeech folders + the LibriSpeechMix JSONL annotation files).
    save_folder:
        The path to the folder where the data manifest JSON files will be stored.
        Default to ``data_folder``.
    splits:
        The dataset splits to load.
        Splits with the same prefix are merged into a single JSON file
        (e.g. "dev-clean-1mix" and "dev-clean-2mix").
        Default to all the available splits.
    num_targets:
        The maximum number of target utterances to extract from each mixture.
        If a list, only the utterances at the given indices are extracted.
        If `min`, use only the shortest utterance as a target.
        If `max`, use only the shortest utterance as a target.
        Default to all the available utterances (`n` for the `n`-mix case).
    num_enrolls:
        The maximum number of enrollment utterances per target speaker.
        Default to all the available enrollment utterances.
    trim_nontarget:
        The maximum number of seconds before and after the target utterance.
        Set to 0 to trim the mixture at the edges of the target utterance.
        Default to infinity.
    suppress_delay:
        True to set all delays to 0 (i.e. maximize the overlap).
        Must be None if `overlap_ratio` is set.
        Default to False.
    overlap_ratio:
        The overlap ratio for the target utterance.
        The target utterance delay is always set to 0, which implies
        that a new mixture is created for each target utterance.
        Must be None if `suppress_delay` is set.
        Default the values specifies in the annotation file for each mixture.

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If the data folder's structure does not match the expected one.

    Examples
    --------
    >>> prepare_librispeechmix(
    ...     "LibriSpeechMix",
    ...     splits=["train-2mix", "dev-clean-2mix", "test-clean-2mix"]
    ... )

    """
    if not save_folder:
        save_folder = data_folder
    if not splits:
        raise ValueError(f"`splits` ({splits}) must be non-empty")
    if suppress_delay is not None and overlap_ratio is not None:
        raise ValueError(
            f"Either `suppress_delay` or `overlap_ratio` must be set, but not both"
        )
    if overlap_ratio is not None:
        if overlap_ratio < 0.0 or overlap_ratio > 1.0:
            raise ValueError(
                f"`overlap_ratio` ({overlap_ratio}) must be in the interval [0, 1]"
            )

    # Grouping
    groups = defaultdict(list)
    for split in splits:
        if split.startswith("train"):
            groups["train"].append(split)
        elif split.startswith("dev"):
            groups["dev"].append(split)
        elif split.startswith("test"):
            groups["test"].append(split)
        else:
            raise ValueError(
                f'`split` ({split}) must start with either "train", "dev" or "test"'
            )

    # Write output JSON for each group
    for group_name, group in groups.items():
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )

        output_entries = {}
        for split in group:
            _LOGGER.info(f"Split: {split}")

            # Read input JSONL
            input_jsonl = os.path.join(data_folder, f"{split}.jsonl")
            if not os.path.exists(input_jsonl):
                raise RuntimeError(f'"{input_jsonl}" not found')
            with open(input_jsonl, "r", encoding="utf-8") as fr:
                for input_line in fr:
                    input_entry = json.loads(input_line)
                    ID = input_entry["id"]
                    wavs = input_entry["wavs"]
                    durations = copy.deepcopy(input_entry["durations"])
                    speaker_profile = input_entry["speaker_profile"]
                    texts = input_entry["texts"]
                    speaker_profile_index = input_entry["speaker_profile_index"]
                    speakers = input_entry["speakers"]
                    genders = input_entry["genders"]

                    if isinstance(num_targets, list):
                        target_speaker_idxes = num_targets
                    elif num_targets == "min":
                        min_duration = min(durations)
                        min_idx = durations.index(min_duration)
                        target_speaker_idxes = [min_idx]
                    elif num_targets == "max":
                        max_duration = max(durations)
                        max_idx = durations.index(max_duration)
                        target_speaker_idxes = [max_idx]
                    else:
                        target_speaker_idxes = list(range(len(texts)))

                    wavs = [os.path.join("{DATA_ROOT}", wav) for wav in wavs]
                    for target_speaker_idx in target_speaker_idxes:
                        text = texts[target_speaker_idx]
                        idx = speaker_profile_index[target_speaker_idx]
                        ID_text = f"{ID}_text-{target_speaker_idx}"

                        # Read here to not overwrite
                        delays = copy.deepcopy(input_entry["delays"])

                        if suppress_delay:
                            delays = [0.0 for _ in delays]

                        if overlap_ratio is not None:
                            target_duration = durations[target_speaker_idx]
                            overlap_start = (1 - overlap_ratio) * target_duration
                            delays = [overlap_start] * len(wavs)
                            delays[target_speaker_idx] = 0

                        start = 0.0
                        duration = max([d + x for d, x in zip(delays, durations)])
                        max_duration = copy.deepcopy(duration)
                        if trim_nontarget is not None:
                            start = delays[target_speaker_idx]
                            duration = durations[target_speaker_idx]
                            new_start = max(0.0, start - trim_nontarget)
                            duration += start - new_start
                            duration = min(
                                duration + trim_nontarget, max_duration - new_start
                            )

                        enroll_wavs = speaker_profile[idx]
                        for enroll_wav in enroll_wavs[:num_enrolls]:
                            ID_enroll = f"{ID_text}_{enroll_wav}"
                            enroll_wav = os.path.join("{DATA_ROOT}", enroll_wav)
                            output_entry = {
                                "wavs": wavs,
                                "enroll_wav": enroll_wav,
                                "delays": delays,
                                "start": start,
                                "duration": duration,
                                "durations": durations,
                                "target_speaker_idx": target_speaker_idx,
                                "wrd": text,
                                "speakers": speakers,
                                "genders": genders,
                            }
                            output_entries[ID_enroll] = output_entry

        # Write output JSON
        output_json = os.path.join(save_folder, f"{group_name}.json")
        _LOGGER.info(f"Writing {output_json}...")
        with open(output_json, "w", encoding="utf-8") as fw:
            json.dump(output_entries, fw, ensure_ascii=False, indent=4)

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")

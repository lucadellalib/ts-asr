"""LibriSpeechMix data preparation.

Authors
 * Luca Della Libera 2023
"""

import json
import logging
import os
from collections import defaultdict
from typing import Optional, Sequence

import torchaudio


__all__ = ["prepare_librispeechmix"]


# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
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
    max_enrolls: "Optional[int]" = None,
) -> "None":
    """Prepare data manifest JSON files for LibriSpeechMix dataset
    (see https://github.com/NaoyukiKanda/LibriSpeechMix).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data
        manifest JSON files will be stored.
        Default to ``data_folder``.
    splits:
        The dataset splits to load.
        Splits with the same prefix are merged into a single
        JSON file (e.g. "dev-clean-1mix" and "dev-clean-2mix").
        Default to all the available splits.
    max_enrolls:
        The maximum number of enrollment utterances per target speaker.
        Default to all the available enrollment utterances.

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
            input_jsonl = os.path.join(data_folder, "list", f"{split}.jsonl")
            if not os.path.exists(input_jsonl):
                raise RuntimeError(
                    f'"{input_jsonl}" not found. Download the data generation '
                    f"scripts from https://github.com/NaoyukiKanda/LibriSpeechMix "
                    f"and follow the readme to generate the data"
                )
            with open(input_jsonl, "r", encoding="utf-8") as fr:
                for input_line in fr:
                    input_entry = json.loads(input_line)
                    ID = input_entry["id"]
                    mixed_wav = input_entry["mixed_wav"]
                    texts = input_entry["texts"]
                    speaker_profile = input_entry["speaker_profile"]
                    speaker_profile_index = input_entry["speaker_profile_index"]
                    # wavs = input_entry["wavs"]
                    # delays = input_entry["delays"]
                    # speakers = input_entry["speakers"]
                    # durations = input_entry["durations"]
                    # genders = input_entry["genders"]

                    info = torchaudio.info(os.path.join(data_folder, "data", mixed_wav))
                    duration = info.num_frames / info.sample_rate

                    mixed_wav = os.path.join("{DATA_ROOT}", "data", mixed_wav)
                    for i, (text, idx) in enumerate(zip(texts, speaker_profile_index)):
                        ID_text = f"{ID}_text-{i}"
                        enroll_wavs = speaker_profile[idx]
                        for enroll_wav in enroll_wavs[:max_enrolls]:
                            ID_enroll = f"{ID_text}_{enroll_wav}"
                            enroll_wav = os.path.join("{DATA_ROOT}", "data", enroll_wav)
                            output_entry = {
                                "mixed_wav": mixed_wav,
                                "enroll_wav": enroll_wav,
                                "wrd": text,
                                "duration": duration,
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

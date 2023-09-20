#!/usr/bin/env/python

"""Plot SpeechBrain WER report against data statistics from a manifest file.

To run this script, do the following:
> python plot_wer_report.py wer.txt

Authors
 * Luca Della Libera 2023
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Tuple

from matplotlib import pyplot as plt, rc


__all__ = [
    "parse_data_manifest",
    "parse_wer_report",
]


def parse_wer_report(wer_report: "str") -> "Dict[str, float]":
    """Parse WER report to extract utterance IDs and corresponding WER values.

    Arguments
    ---------
    wer_report:
        The path to the WER report file.

    Returns
    -------
        The WERs, i.e. a dict that maps utterance IDs
        to the corresponding WER values.

    Examples
    --------
    >>> wers = parse_wer_report("wer_test-clean-2mix.txt")

    """
    wers = {}
    with open(wer_report) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(",", 1)
            if len(tokens) != 2:
                continue
            utterance_id, tail = tokens
            if not tail.endswith("]"):
                continue
            pattern = r"%WER\s+(\d+\.\d+)"
            match = re.search(pattern, tail)
            if not match:
                continue
            wer = float(match.group(1))
            wers[utterance_id] = wer
    return wers


def parse_data_manifest(data_manifest: "str") -> "Dict[str, Dict]":
    """Parse data manifest to extract utterance IDs and corresponding feature values.

    Arguments
    ---------
    data_manifest:
        The path to the data manifest file.

    Returns
    -------
        The features, i.e. a dict that maps utterance IDs
        to the corresponding feature values.

    Examples
    --------
    >>> features = parse_data_manifest("test.json")

    """
    features = defaultdict(dict)
    if data_manifest.endswith(".json"):
        with open(data_manifest, "r", encoding="utf-8") as fr:
            data = json.load(fr)
    for utterance_id, entry in data.items():
        target_speaker_idx = entry["target_speaker_idx"]
        # wav = datum["wavs"][target_speaker_idx]
        # enroll_wav = datum["enroll_wav"]
        delays = entry["delays"]
        start = entry["start"]
        duration = entry["duration"]
        durations = entry["durations"]
        # text = datum["wrd"]
        # speakers = datum["speakers"]
        # genders = datum["genders"]

        # Duration
        features[utterance_id]["Duration (s)"] = duration

        # Target duration
        target_start = delays[target_speaker_idx]
        target_end = target_start + durations[target_speaker_idx]
        target_start = max(target_start, start)
        target_end = min(target_end, start + duration)
        target_duration = target_end - target_start
        features[utterance_id]["Target duration (s)"] = target_duration

        # Interference duration
        interference_endpoints = []
        for i, interference_start in enumerate(delays):
            if i == target_speaker_idx:
                continue
            interference_end = interference_start + durations[i]
            interference_start = max(interference_start, start)
            interference_end = min(interference_end, start + duration)
            if interference_end - interference_start <= 0:
                continue
            interference_endpoints.append((interference_start, interference_end))
        interference_endpoints = sorted(interference_endpoints, key=lambda x: x[0])
        interference_segments = [interference_endpoints[0]]
        for interference_start, interference_end in interference_endpoints[1:]:
            last_interference_segment = interference_segments[-1]
            if interference_start <= last_interference_segment[1]:
                interference_segments.pop(-1)
                interference_segments.append(
                    (last_interference_segment[0], interference_end)
                )
            else:
                interference_segments.append((interference_start, interference_end))
        interference_durations = [x[1] - x[0] for x in interference_segments]
        interference_duration = sum(interference_durations)
        features[utterance_id]["Interference duration (s)"] = interference_duration

        # Overlap ratio
        interference_segments_within_target = []
        for interference_segment in interference_segments:
            interference_segment_within_target = (
                max(interference_segment[0], target_start),
                min(interference_segment[1], target_end),
            )
            if (
                interference_segment_within_target[1]
                - interference_segment_within_target[0]
                <= 0
            ):
                continue
            interference_segments_within_target.append(
                interference_segment_within_target
            )
        interference_durations_within_target = [
            x[1] - x[0] for x in interference_segments_within_target
        ]
        interference_duration_within_target = sum(interference_durations_within_target)
        overlap_ratio = (interference_duration_within_target / target_duration) * 100
        features[utterance_id]["Overlap ratio (%)"] = overlap_ratio

    return features


def plot_wers(
    wers: "Dict[str, float]",
    features: "Dict[str, float]",
    output_image: "str",
    xlabel,
    title: "str" = "",
    figsize: "Tuple[float, float]" = (8.0, 4.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot WERs extracted from a WER report file.

    Arguments
    ---------
    wers:
        The WERs, i.e. a dict that maps utterance IDs
        to the corresponding WER values.
    features:
        The features, i.e. a dict that maps utterance IDs
        to the corresponding feature values.
    output_image:
        The path to the output image.
    xlabel:
        The x-axis label.
    title:
        The title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one
        of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> wers = parse_wer_report("wer_test-clean-2mix.txt")
    >>> plot_wers(wers, features, "wer_test-clean-2mix.png")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        rc("text", usetex=usetex)
        fig = plt.figure(figsize=figsize)
        utterance_ids = list(wers.keys())
        wers = [wers[x] for x in utterance_ids]
        features = [features[x] for x in utterance_ids]
        plt.scatter(features, wers)
        plt.grid()
        plt.title(title)
        plt.xlabel(xlabel)
        # xmin, xmax = plt.xlim()
        # xrange = xmax - xmin
        # plt.xlim(xmin - 0.025 * xrange, xmax + 0.025 * xrange)
        plt.ylabel("WER (%)")
        # ymin, ymax = plt.ylim()
        # yrange = ymax - ymin
        # plt.ylim(-0.1)
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot WER report against data statistics from a manifest file"
    )
    parser.add_argument(
        "wer_report", help="path to WER report",
    )
    parser.add_argument(
        "data_manifest", help="path to data manifest",
    )
    parser.add_argument(
        "feature", help="feature to plot along the x-axis",
    )
    parser.add_argument(
        "-o", "--output_image", help="path to output image",
    )
    parser.add_argument(
        "-x", "--xlabel", help="x-axis label",
    )
    parser.add_argument(
        "-t", "--title", help="title",
    )
    parser.add_argument(
        "-f", "--figsize", nargs=2, default=(8.0, 4.0), help="figure size",
    )
    parser.add_argument(
        "-u", "--usetex", action="store_true", help="render text with LaTeX",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="classic",
        help="path to a Matplotlib style file or name of one of Matplotlib built-in styles",
        dest="style_file_or_name",
    )
    args = parser.parse_args()
    wers = parse_wer_report(args.wer_report)
    features = parse_data_manifest(args.data_manifest)
    feature = args.feature
    output_image = (
        args.output_image
        or args.wer_report.replace(".txt", "") + f"_{feature}" + ".png"
    )
    xlabel = args.xlabel or feature
    title = args.title or args.wer_report
    plot_wers(
        wers,
        {x: features[x][feature] for x in features},
        output_image,
        xlabel,
        title,
        args.figsize,
        args.usetex,
        args.style_file_or_name,
    )

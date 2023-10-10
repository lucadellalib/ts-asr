#!/usr/bin/env/python

"""Plot data statistics from a JSON manifest file.

To run this script (requires Matplotlib installed):
> python plot_data.py <path-to-manifest.json> <feature>

Authors
 * Luca Della Libera 2023
"""

import argparse
import contextlib
import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union


try:
    from matplotlib import pyplot as plt, rc
except ImportError:
    raise ImportError("This script requires Matplotlib (`pip install matplotlib`)")


@contextlib.contextmanager
def _set_style(style_file_or_name="classic", usetex=False, fontsize=12):
    """Set plotting style.

    Arguments
    ---------
    style_file_or_name : str, optional
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    usetex : bool, optional
        True to render text with LaTeX, False otherwise.
    fontsize : int, optional
        The global font size.

    """
    # Customize style
    try:
        plt.style.use(style_file_or_name)
        plt.rc("font", size=fontsize)
        plt.rc("axes", titlesize=fontsize)
        plt.rc("axes", labelsize=fontsize)
        plt.rc("xtick", labelsize=fontsize - 1)
        plt.rc("ytick", labelsize=fontsize - 1)
        plt.rc("legend", fontsize=fontsize)
        plt.rc("figure", titlesize=fontsize)
        rc("text", usetex=usetex)
        if usetex:
            rc("font", family="serif", serif=["Computer Modern"])
        yield
    finally:
        plt.style.use("default")


def parse_data_manifest(data_manifest: "str") -> "Dict[str, Dict]":
    """Parse data JSON manifest to extract utterance IDs and
    corresponding feature values.

    The following features are available:
    - "duration"
    - "target_duration"
    - "nontarget_duration"
    - "interference_duration"
    - "overlap_ratio_target"
    - "overlap_ratio_mixture"

    Arguments
    ---------
    data_manifest:
        The path to the data manifest file.

    Returns
    -------
        The features, i.e. a dict that maps utterance
        IDs to the corresponding feature values.

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
        features[utterance_id]["duration"] = duration

        # Target duration
        target_start = delays[target_speaker_idx]
        target_end = target_start + durations[target_speaker_idx]
        target_start = max(target_start, start)
        target_end = min(target_end, start + duration)
        target_duration = target_end - target_start
        features[utterance_id]["target_duration"] = target_duration

        # Non-target duration
        features[utterance_id]["nontarget_duration"] = duration - target_duration

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
        features[utterance_id]["interference_duration"] = interference_duration

        # Overlap ratio with respect to the target
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
        overlap_ratio_target = (
            interference_duration_within_target / target_duration
        ) * 100
        features[utterance_id]["overlap_ratio_target"] = overlap_ratio_target

        # Overlap ratio with respect to the mixture
        mixture_duration = max(x + y for x, y in zip(durations, delays))
        overlap_ratio_mixture = (
            interference_duration_within_target / mixture_duration
        ) * 100
        features[utterance_id]["overlap_ratio_mixture"] = overlap_ratio_mixture

    return features


def plot_data(
    features: "Dict[str, Union[float, Tuple[float, float]]]",
    output_image: "str",
    xlabel: "Optional[str]" = None,
    ylabel: "Optional[str]" = None,
    title: "Optional[str]" = None,
    figsize: "Tuple[float, float]" = (6.0, 4.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot data statistics extracted from a data manifest file.

    Arguments
    ---------
    features:
        The features, i.e. a dict that maps utterance
        IDs to the corresponding feature values.
        If the feature value is a pair, a scatter plot is produced.
        Otherwise, a histogram is produced.
    output_image:
        The path to the output image.
    xlabel:
        The x-axis label.
    ylabel:
        The y-axis label.
    title:
        The title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> features = parse_data_manifest("test.json")
    >>> plot_data({x: features[x]["duration"] for x in features}, "test.jpg")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with _set_style(style_file_or_name, usetex):
        plt.figure(figsize=figsize)
        features = list(features.values())
        if isinstance(features[0], (tuple, list)):
            x_features, y_features = zip(*features)
            plt.scatter(x_features, y_features)
        else:
            plt.hist(features, histtype="step")
        plt.grid()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        else:
            plt.title(f"{len(features)} samples")
        # xmin, xmax = plt.xlim()
        # xrange = xmax - xmin
        # plt.xlim(xmin - 0.025 * xrange, xmax + 0.025 * xrange)
        # ymin, ymax = plt.ylim()
        # yrange = ymax - ymin
        # plt.ylim(ymin - 0.025 * yrange, ymax + 0.025 * yrange)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot data statistics from a JSON manifest file"
    )
    parser.add_argument(
        "data_manifest", help="path to data manifest",
    )
    parser.add_argument(
        "feature",
        nargs="+",
        help="feature(s) to plot",
        choices=[
            "duration",
            "target_duration",
            "nontarget_duration",
            "interference_duration",
            "overlap_ratio_target",
            "overlap_ratio_mixture",
        ],
    )
    parser.add_argument(
        "-o", "--output_image", help="path to output image",
    )
    parser.add_argument(
        "-x", "--xlabel", help="x-axis label",
    )
    parser.add_argument(
        "-y", "--ylabel", help="y-axis label",
    )
    parser.add_argument(
        "-t", "--title", help="title",
    )
    parser.add_argument(
        "-f", "--figsize", nargs=2, default=(6.0, 4.0), help="figure size",
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
    features = parse_data_manifest(args.data_manifest)
    feature = args.feature

    if len(feature) == 1:
        feature = feature[0]
        features = {x: features[x][feature] for x in features}
        output_image = (
            args.output_image
            or args.data_manifest.replace(".json", "") + f"_{feature}" + ".jpg"
        )
        xlabel = args.xlabel or feature
        ylabel = args.ylabel or "Count"
    elif len(feature) == 2:
        features = {
            x: (features[x][feature[0]], features[x][feature[1]]) for x in features
        }
        output_image = (
            args.output_image
            or args.data_manifest.replace(".json", "")
            + f"_{feature[0]}_vs_{feature[1]}"
            + ".jpg"
        )
        xlabel = args.xlabel or feature[0]
        ylabel = args.ylabel or feature[1]
    else:
        raise ValueError("No more than 2 features can be provided")

    title = args.title
    plot_data(
        features,
        output_image,
        xlabel,
        ylabel,
        title,
        args.figsize,
        args.usetex,
        args.style_file_or_name,
    )

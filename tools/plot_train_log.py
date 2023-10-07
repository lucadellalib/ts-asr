#!/usr/bin/env/python

"""Plot train log.

To run this script (requires Matplotlib installed):
> python plot_train_log.py <path-to-train_log.txt>

Authors
 * Luca Della Libera 2023
"""

import argparse
import contextlib
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
from numpy import ndarray


try:
    from matplotlib import pyplot as plt, rc
except ImportError:
    raise ImportError("This script requires Matplotlib (`pip install matplotlib`)")


_EXPECTED_METRICS = [
    "epoch",
    "train loss",
    "valid loss",
    "valid CER",
    "valid WER",
]


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


def parse_train_log(train_log: "str") -> "Dict[str, ndarray]":
    """Parse train log to extract metric names and values.

    Arguments
    ---------
    train_log:
        The path to the train log file.

    Returns
    -------
        The metrics, i.e. a dict that maps names of
        the metrics to the metric values themselves.

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")

    """
    metrics = defaultdict(list)
    with open(train_log) as f:
        for line in f:
            line = line.strip().replace(" - ", ", ")
            if not line:
                continue
            tokens = line.split(", ")
            names, values = zip(*[token.split(": ") for token in tokens])
            names, values = list(names), list(values)
            for name in _EXPECTED_METRICS:
                if name not in names:
                    names.append(name)
                    values.append("nan")
            for name, value in zip(names, values):
                try:
                    metrics[name].append(float(value))
                except Exception:
                    pass
    for name, values in metrics.items():
        metrics[name] = np.array(values)
    return metrics


def plot_metrics(
    metrics: "Dict[str, ndarray]",
    output_image: "str",
    xlabel: "Optional[str]" = None,
    ylabel: "Optional[str]" = None,
    title: "Optional[str]" = None,
    figsize: "Tuple[float, float]" = (6.0, 4.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot metrics extracted from train log.

    Arguments
    ---------
    metrics:
        The metrics, i.e. a dict that maps names of
        the metrics to the metric values themselves.
    output_image:
        The path to the output image.
    xlabel:
        The x-axis label.
    ylabel:
        The y-axis label.
    title:
        The title.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")
    >>> plot_metrics(metrics, "train_log.jpg")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with _set_style(style_file_or_name, usetex):
        # Train
        plt.plot(
            metrics["epoch"], metrics["train loss"], marker="o", label="Train loss",
        )

        # Validation
        plt.plot(
            metrics["epoch"],
            metrics["valid loss"],
            marker="X",
            label="Validation loss",
        )
        for i, value in enumerate(metrics["valid WER"]):
            if i % 10 == 0:
                plt.annotate(
                    f"WER={value}%", (i + 1, metrics["valid loss"][i]),
                )

        # Test
        if "test_loss" in metrics:
            plt.plot(
                metrics["epoch"], metrics["test loss"], marker="D", label="Test loss",
            )
            for i, value in enumerate(metrics["test WER"]):
                plt.annotate(
                    f"WER={value}%", (i + 1, metrics["test loss"][i]),
                )

        plt.grid()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        # xmin, xmax = plt.xlim()
        # xrange = xmax - xmin
        # plt.xlim(xmin - 0.025 * xrange, xmax + 0.025 * xrange)
        # ymin, ymax = plt.ylim()
        # yrange = ymax - ymin
        # plt.ylim(ymin - 0.025 * yrange, ymax + 0.025 * yrange)
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train log")
    parser.add_argument(
        "train_log", help="path to train log",
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
        "-f", "--figsize", nargs=2, default=(6.0, 4.0), type=float, help="figure size",
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
    metrics = parse_train_log(args.train_log)
    output_image = args.output_image or args.train_log.replace(".txt", ".jpg")
    xlabel = args.xlabel or "Epoch"
    ylabel = args.ylabel or "Loss"
    title = args.title
    plot_metrics(
        metrics,
        output_image,
        xlabel,
        ylabel,
        title,
        args.figsize,
        args.usetex,
        args.style_file_or_name,
    )

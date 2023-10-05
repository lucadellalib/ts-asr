"""Common utilities.

Authors
* Luca Della Libera 2023
"""

import logging
import os
import subprocess

import numpy as np
from scipy.io.wavfile import write


__all__ = [
    "play_waveform",
    "plot_attentions",
    "plot_embeddings",
    "plot_spectrogram",
    "plot_waveform",
]


def play_waveform(waveform, sample_rate, output_file="waveform.wav", interactive=False):
    """Play a waveform (requires FFplay installed on the system).

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [num_channels, num_frames].
    sample_rate : int
        The sample rate.
    output_file : str, optional
        The path to the output image.
    interactive : bool, optional
        True to plot interactively, False otherwise.

    """
    waveform = np.array(waveform)
    if waveform.ndim == 1:
        waveform = waveform[None]

    write(output_file, sample_rate, np.transpose(waveform))
    if interactive:
        subprocess.call(["ffplay", output_file])


def plot_waveform(
    waveform,
    sample_rate,
    output_image="waveform.png",
    labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(4.0, 4.0),
    usetex=False,
    legend=False,
    style_file_or_name="classic",
    interactive=False,
):
    """Plot a waveform in the time domain.

    Arguments
    ---------
    waveform : np.ndarray or list
        The raw waveform(s), shape: [num_frames].
    sample_rate : int
        The sample rate.
    output_image : str, optional
        The path to the output image.
    labels: list, optional
        The label for each waveform.
        Used only if `waveform` is a list.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    title : str, optional
        The title.
    figsize : tuple, optional
        The figure size.
    usetex : bool, optional
        True to render text with LaTeX, False otherwise.
    legend : bool, optional
        True to show the legend, False otherwise.
    style_file_or_name : str, optional
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    interactive : bool, optional
        True to plot interactively, False otherwise.

    """
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    if isinstance(waveform, (tuple, list)):
        waveforms = [np.array(x).squeeze() for x in waveform]
    else:
        waveforms = [np.array(waveform).squeeze()]
    if labels is None:
        labels = [None] * len(waveforms)

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        # Customize style
        if usetex:
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"])

        plt.figure(figsize=figsize)
        for i, x in enumerate(waveforms):
            num_frames = x.shape[0]
            time_axis = np.arange(num_frames) / sample_rate
            plt.plot(time_axis, x, label=labels[i])
        plt.grid()
        if legend:
            plt.legend(fancybox=True)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if interactive:
            plt.show(block=False)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


def plot_spectrogram(
    waveform,
    sample_rate,
    output_image="spectrogram.png",
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(4.0, 4.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
):
    """Plot a waveform in the time-frequency domain.

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [num_frames].
    sample_rate : int
        The sample rate.
    output_image : str, optional
        The path to the output image.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    title : str, optional
        The title.
    figsize : tuple, optional
        The figure size.
    usetex : bool, optional
        True to render text with LaTeX, False otherwise.
    style_file_or_name : str, optional
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    interactive : bool, optional
        True to plot interactively, False otherwise.

    """
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        # Customize style
        if usetex:
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"])

        plt.figure(figsize=figsize)
        waveform = np.array(waveform).squeeze()
        plt.specgram(waveform, Fs=sample_rate)
        plt.grid()
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if interactive:
            plt.show(block=False)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


def plot_attentions(
    attentions,
    output_image="attentions.png",
    figsize=(16.0, 8.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
):
    """Plot per-layer attention maps.

    Arguments
    ---------
    attentions : np.ndarray
        The per-layer attention maps, shape: [num_layers, num_heads, seq_length, seq_length].
    output_image : str, optional
        The path to the output image.
    figsize : tuple, optional
        The figure size.
    usetex : bool, optional
        True to render text with LaTeX, False otherwise.
    style_file_or_name : str, optional
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    interactive : bool, optional
        True to plot interactively, False otherwise.

    """
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        # Customize style
        if usetex:
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"])

        attentions = np.array(attentions)
        L, H, T, T = attentions.shape
        fig, axes = plt.subplots(L, H, figsize=figsize, squeeze=False)
        for i, layer in enumerate(attentions):
            for j, head in enumerate(layer):
                ax = axes[i, j]
                im = ax.imshow(head, cmap="viridis")
                ax.tick_params(direction="out")
                if i == 0:
                    ax.set_title(f"Head {j + 1}")
                if j == 0:
                    ax.set_ylabel(f"Layer {i + 1}")
                if i < L - 1:
                    ax.set_xticks([], [])
                else:
                    ax.xaxis.tick_bottom()
                if j < H - 1:
                    ax.set_yticks([], [])
                else:
                    ax.yaxis.tick_right()

        if interactive:
            plt.show(block=False)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


def plot_embeddings(
    embeddings,
    labels,
    output_image="embeddings.png",
    xlabel="t-SNE x",
    ylabel="t-SNE y",
    title=None,
    figsize=(4.0, 4.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
    **tsne_kwargs,
):
    """Plot embeddings via 2D t-SNE.

    Arguments
    ---------
    embeddings : np.ndarray
        The embeddings, shape: [num_embeddings, embedding_dim].
    labels : list
        The embedding labels, length: [num_embeddings].
    output_image : str, optional
        The path to the output image.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    title : str, optional
        The title.
    figsize : tuple, optional
        The figure size.
    usetex : bool, optional
        True to render text with LaTeX, False otherwise.
    style_file_or_name : str, optional
        The path to a Matplotlib style file or the name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    interactive : bool, optional
        True to plot interactively, False otherwise.
    tsne_kwargs : dict, optional
        The 2D t-SNE keyword arguments.

    """
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    try:
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.manifold import TSNE
    except ImportError:
        logging.warning(
            "This function requires scikit-learn (`pip install scikit-learn`)"
        )
        return

    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, **tsne_kwargs)
    embeddings = tsne.fit_transform(embeddings)

    if not isinstance(labels[0], int):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        # Customize style
        if usetex:
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"])

        plt.figure(figsize=figsize)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
        plt.grid()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if interactive:
            plt.show(block=False)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()

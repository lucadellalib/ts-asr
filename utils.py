"""Common utilities.

Authors
* Luca Della Libera 2023
"""

import contextlib
import logging
import os
import subprocess

import numpy as np
from scipy.io.wavfile import write


__all__ = [
    "play_waveform",
    "plot_attention",
    "plot_embeddings",
    "plot_fbanks",
    "plot_grad_norm",
    "plot_waveform",
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
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        yield
        return

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


def play_waveform(waveform, sample_rate, output_file="waveform.wav", interactive=False):
    """Play a waveform (requires FFplay installed on the system).

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [num_frames].
    sample_rate : int
        The sample rate.
    output_file : str, optional
        The path to the output file.
    interactive : bool, optional
        True to play interactively, False otherwise.

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
    opacity=1.0,
    output_image="waveform.jpg",
    labels=None,
    xlabel="Time (s)",
    ylabel="Amplitude",
    title=None,
    figsize=(6.0, 4.0),
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
    opacity: float, optional
        The opacity (useful to plot overlapped waveforms).
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
        labels = [f"Waveform {i + 1}" for i in range(len(waveforms))]

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with _set_style(style_file_or_name, usetex):
        plt.figure(figsize=figsize)
        for i, x in enumerate(waveforms):
            num_frames = x.shape[0]
            time_axis = np.arange(num_frames) / sample_rate
            plt.plot(time_axis, x, label=labels[i], alpha=opacity)
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


def plot_fbanks(
    waveform,
    sample_rate,
    output_image="fbanks.jpg",
    xlabel="Feature frame",
    ylabel="Frequency (Hz)",
    title=None,
    figsize=(10.0, 10.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
    **fbanks_kwargs,
):
    """Plot a waveform in the time-frequency domain by
    extracting filter bank features.

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
    fbanks_kwargs : dict, optional
        The filter banks keyword arguments.

    """
    try:
        from matplotlib import pyplot as plt, rc
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    try:
        from speechbrain.lobes.features import Fbank
        import torch
    except ImportError:
        logging.warning(
            "This function requires SpeechBrain (`pip install speechbrain`)"
        )
        return

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with _set_style(style_file_or_name, usetex):
        plt.figure(figsize=figsize)
        waveform = torch.as_tensor(waveform).squeeze()[None]
        if not fbanks_kwargs:
            fbanks_kwargs = {"n_fft": 512, "n_mels": 80, "win_length": 32}
        fbank = Fbank(sample_rate=sample_rate, **fbanks_kwargs)
        fbanks = np.array(fbank(waveform)[0].T)
        plt.imshow(fbanks, origin="lower")
        yrange = np.arange(0, fbanks_kwargs["n_mels"] + 1, 20)
        plt.yticks(yrange)
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


def plot_attention(
    attention,
    average=True,
    output_image="attention.jpg",
    xlabel="Feature frame",
    ylabel="Feature frame",
    figsize=(4.0, 4.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
):
    """Plot an attention map.

    Arguments
    ---------
    attention : np.ndarray
        The attention map, shape: [num_heads, query_length, key_value_length].
    average : bool, optional
        True to average the attention heads, False otherwise.
    output_image : str, optional
        The path to the output image.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
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
    with _set_style(style_file_or_name, usetex):
        attention = np.array(attention)
        if average:
            attention = attention.mean(axis=0, keepdims=True)
        H = attention.shape[0]
        # fig, axes = plt.subplots(
        #    1, H + 1, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]}
        # )
        fig, axes = plt.subplots(1, H, figsize=figsize, squeeze=False)
        for i, head in enumerate(attention):
            ax = axes[0, i]
            ax.imshow(head, cmap="viridis")
            # im = ax.imshow(head, cmap="viridis")
            if H == 1:
                ax.set_title("Average attention")
            else:
                ax.set_title(f"Head {i + 1}")
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                if i == 0:
                    ax.set_ylabel(ylabel)
        # plt.colorbar(im, axes[-1], shrink=0.75)

        if interactive:
            plt.show(block=False)
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


def plot_embeddings(
    embeddings,
    labels,
    output_image="embeddings.jpg",
    xlabel="t-SNE x",
    ylabel="t-SNE y",
    title=None,
    figsize=(6.0, 4.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
    **tsne_kwargs,
):
    """Plot embeddings via 2D t-SNE.

    Arguments
    ---------
    embeddings : np.ndarray or list
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
    with _set_style(style_file_or_name, usetex):
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


def plot_grad_norm(
    grad_norm,
    output_image="grad_norm.jpg",
    xlabel="Epoch",
    ylabel="Gradient L2 norm",
    title=None,
    figsize=(6.0, 4.0),
    usetex=False,
    style_file_or_name="classic",
    interactive=False,
):
    """Plot the gradient norm.

    Arguments
    ---------
    grad_norm : np.ndarray
        The gradient norm.
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

    try:
        from speechbrain.lobes.features import Fbank
        import torch
    except ImportError:
        logging.warning(
            "This function requires SpeechBrain (`pip install speechbrain`)"
        )
        return

    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with _set_style(style_file_or_name, usetex):
        plt.figure(figsize=figsize)
        grad_norm = np.array(grad_norm).squeeze()
        plt.plot(range(1, len(grad_norm) + 1), grad_norm)
        plt.xlim(1, len(grad_norm))
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

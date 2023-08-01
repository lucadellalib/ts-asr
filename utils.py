"""Common utilities.

Authors
* Luca Della Libera 2023
"""

import logging
import subprocess

import numpy as np
from scipy.io.wavfile import write


__all__ = [
    "play_waveform",
    "plot_spectrogram",
    "plot_waveform",
]


def play_waveform(waveform, sample_rate):
    """Play a waveform (requires FFplay installed on the system).

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [(num_channels), num_frames].
    sample_rate : int
        The sample rate.

    """
    if waveform.ndim == 1:
        waveform = waveform[None]

    write("waveform.wav", sample_rate, np.transpose(waveform))
    subprocess.call(["ffplay", "waveform.wav"])


def plot_waveform(waveform, sample_rate, title="Waveform"):
    """Plot a waveform.

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [(num_channels), num_frames].
    sample_rate : int
        The sample rate.
    title : str, optional
        The plot title.

    """
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    if waveform.ndim == 1:
        waveform = waveform[None]

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for channel in range(num_channels):
        axes[channel].plot(time_axis, waveform[channel], linewidth=1)
        axes[channel].grid(True)
        if num_channels > 1:
            axes[channel].set_ylabel(f"Channel {channel + 1}")
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(waveform, sample_rate=16000, title="Spectrogram"):
    """Plot a waveform spectrogram.

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [(num_channels), num_frames].
    sample_rate : int
        The sample rate.
    title : str, optional
        The plot title.

    """
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        logging.warning("This function requires Matplotlib (`pip install matplotlib`)")
        return

    if waveform.ndim == 1:
        waveform = waveform[None]

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for channel in range(num_channels):
        axes[channel].specgram(waveform[channel], Fs=sample_rate)
        if num_channels > 1:
            axes[channel].set_ylabel(f"Channel {channel + 1}")
    figure.suptitle(title)
    plt.show(block=False)

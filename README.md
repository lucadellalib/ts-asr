# Target Speaker Automatic Speech Recognition

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10 | 3.11](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11-blue)](https://www.python.org/downloads/)

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train end-to-end transducer-based target speaker automatic speech recognition
(TS-ASR) systems as proposed in [Streaming Target-Speaker ASR with Neural Transducer](https://arxiv.org/abs/2209.04175)
on [LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix).

---------------------------------------------------------------------------------------------------------

## ⚡ Dataset

Generate the LibriSpeechMix data in `<path-to-data-folder>` following the
[official readme](https://github.com/NaoyukiKanda/LibriSpeechMix/blob/main/README.md).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Clone the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<path-to-repository>`, open a terminal and run:

```bash
python train.py hparams/<config>.yaml --data_folder <path-to-data-folder>
```

To use multiple GPUs available on the same node, run:

```bash
python -m torch.distributed.launch --nproc_per_node=<num-gpus> \
train.py hparams/<config>.yaml --data_folder <path-to-data-folder> --distributed_launch
```

**NOTE**: a single GPU is used for inference, even if multiple GPUs are available.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------

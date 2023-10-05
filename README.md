# Target Speaker Automatic Speech Recognition

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10 | 3.11](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11-blue)](https://www.python.org/downloads/)

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train end-to-end transducer-based target speaker automatic
speech recognition (TS-ASR) systems as proposed in [Streaming Target-Speaker ASR with Neural Transducer](https://arxiv.org/abs/2209.04175).

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### LibriSpeechMix

Generate the LibriSpeechMix data in `<path-to-data-folder>` following the
[official readme](https://github.com/NaoyukiKanda/LibriSpeechMix/blob/main/README.md).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Clone the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
pip install -e vendor/speechbrain
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<path-to-repository>`, open a terminal and run:

```bash
python train_<dataset>_<variant>.py hparams/<dataset>/<config>.yaml --data_folder <path-to-data-folder>
```

To use multiple GPUs on the same node, run:

```bash
python -m torch.distributed.launch --nproc_per_node=<num-gpus> \
train_<dataset>_<variant>.py hparams/<dataset>/<config>.yaml --data_folder <path-to-data-folder> --distributed_launch
```

To use multiple GPUs on multiple nodes, for each node with rank `0, ..., <num-nodes> - 1` run:

```bash
python -m torch.distributed.launch --nproc_per_node=<num-gpus-per-node> \
--nnodes=<num-nodes> --node_rank=<node-rank> --master_addr <rank-0-ip-addr> --master_port 5555 \
train_<dataset>_<variant>.py hparams/<dataset>/<config>.yaml --data_folder <path-to-data-folder> --distributed_launch
```

Helper functions and scripts for plotting and analyzing the results can be found in `utils.py` and `tools`.

**NOTE**: the vendored version of SpeechBrain inside this repository includes several hotfixes (e.g. distributed training,
gradient clipping, gradient accumulation, causality, etc.) and additional features (e.g. distributed evaluation).

### Examples

```bash
nohup python -m torch.distributed.launch --nproc_per_node=8 \
train_librispeechmix_scratch.py hparams/LibriSpeechMix/conformer-t_scratch.yaml \
--data_folder datasets/LibriSpeechMix --num_epochs 100 \
--distributed_launch &
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------

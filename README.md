# Target Speaker Automatic Speech Recognition

This recipe includes scripts to train transducer-based target speaker automatic speech recognition]
(TS-ASR) systems (see https://arxiv.org/abs/2209.04175) on [LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix).

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

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------

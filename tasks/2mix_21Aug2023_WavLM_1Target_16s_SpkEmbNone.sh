# Instance type: p3.16xlarge
# Activate virtual environment: conda activate pytorch_p38

ROOT_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
DATA_DIR=/efs-storage

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=8 \
train_librispeechmix_pretrained.py \
hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/2mix_21Aug2023_WavLM_1Target_16s_SpkEmbNone \
--num_epochs 100 \
--augment True \
--num_targets 1 \
--train_remove_if_longer 16.0 \
--valid_remove_if_longer 16.0 \
--test_remove_if_longer 16.0 \
--injection_mode null \
--find_unused_parameters \
--distributed_launch

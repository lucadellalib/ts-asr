# Instance type: p3.16xlarge
# Activate virtual environment: conda activate pytorch_p38

ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")/../../
DATA_DIR=/efs-storage

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=8 \
train_librispeechmix_pretrained.py \
hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/wavlm/2mix_21Aug2023_WavLM_20s_SpkEmbCat07 \
--num_epochs 50 \
--augment True \
--train_remove_if_longer 20.0 \
--injection_mode cat \
--injection_after [0, 7] \
--distributed_launch

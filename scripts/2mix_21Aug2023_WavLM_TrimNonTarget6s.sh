# Instance type: p3.16xlarge

ROOT_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
DATA_DIR=/efs-storage

python -m torch.distributed.launch --nproc_per_node=8 \
$ROOT_DIR/train_librispeechmix_pretrained.py \
$ROOT_DIR/hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder $ROOT_DIR/results/2mix_21Aug2023_WavLM_TrimNonTarget6s \
--num_epochs 100 \
--augment True \
--trim_nontarget 6.0 \
--distributed_launch

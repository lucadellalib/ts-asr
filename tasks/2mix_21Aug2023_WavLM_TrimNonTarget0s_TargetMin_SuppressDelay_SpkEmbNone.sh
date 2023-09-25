# Instance type: p3.16xlarge
# Activate virtual environment: conda activate pytorch_p38

ROOT_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
DATA_DIR=/efs-storage

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=8 \
train_librispeechmix_pretrained.py \
hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/2mix_21Aug2023_WavLM_TrimNonTarget0s_TargetMin_SuppressDelay_SpkEmbNone \
--num_epochs 40 \
--augment True \
--trim_nontarget 0.0 \
--num_targets min \
--suppress_delay True \
--injection_mode null \
--distributed_launch

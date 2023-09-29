# Instance type: p3.16xlarge
# Activate virtual environment: conda activate pytorch_p38

ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")/../../
DATA_DIR=/efs-storage

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=8 \
train_librispeechmix_pretrained.py \
hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/wavlm/2mix_21Aug2023_WavLM_Overlap20_SpkEmbCat \
--num_epochs 50 \
--augment True \
--overlap_ratio 0.2 \
--injection_mode cat \
--distributed_launch

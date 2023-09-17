ROOT_DIR=../
DATA_DIR=/efs-storage

python -m torch.distributed.launch --nproc_per_node=8 \
$ROOT_DIR/train_librispeechmix.py \
$ROOT_DIR/hparams/LibriSpeechMix/conformer-t.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder $ROOT_DIR/results/2mix_21Aug2023_WavLM_TrimNonTarget0s \
--num_epochs 100 \
--augment True \
--trim_nontarget 0.0 \
--distributed_launch

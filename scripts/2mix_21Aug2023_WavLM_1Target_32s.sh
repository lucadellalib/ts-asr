ROOT_DIR=../
DATA_DIR=/efs-storage

python -m torch.distributed.launch --nproc_per_node=8 \
$ROOT_DIR/train_librispeechmix.py \
$ROOT_DIR/hparams/LibriSpeechMix/conformer-t.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder $ROOT_DIR/results/2mix_21Aug2023_WavLM_1Target_32s \
--num_epochs 100 \
--augment True \
--num_targets 1 \
--train_remove_if_longer 32.0 \
--valid_remove_if_longer 32.0 \
--test_remove_if_longer 32.0 \
--distributed_launch

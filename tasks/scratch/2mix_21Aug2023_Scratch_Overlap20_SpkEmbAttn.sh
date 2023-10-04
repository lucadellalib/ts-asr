# NOTE: activate virtual environment (if any) before running this script

CONFIG=$(dirname "${BASH_SOURCE[0]}")/../config.sh
source $CONFIG

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
train_librispeechmix_scratch.py \
hparams/LibriSpeechMix/conformer-t_scratch.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/scratch/2mix_21Aug2023_Scratch_Overlap20_SpkEmbAttn \
--num_epochs $NUM_EPOCHS \
--augment $AUGMENT \
--overlap_ratio 0.2 \
--injection_mode cross_attention \
--distributed_launch

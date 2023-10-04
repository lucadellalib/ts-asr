# NOTE: activate virtual environment (if any) before running this script

CONFIG=$(dirname "${BASH_SOURCE[0]}")/../config.sh
source $CONFIG

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
train_librispeechmix_none.py \
hparams/LibriSpeechMix/conformer-t_none.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/none/2mix_21Aug2023_None_20s_TrimNonTarget0s \
--num_epochs $NUM_EPOCHS \
--augment $AUGMENT \
--train_remove_if_longer 20.0 \
--trim_nontarget 0.0 \
--distributed_launch

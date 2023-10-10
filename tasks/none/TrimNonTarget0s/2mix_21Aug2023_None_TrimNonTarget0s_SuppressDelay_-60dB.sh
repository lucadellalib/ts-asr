# NOTE: activate virtual environment (if any) before running this script

SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME="${SCRIPT_NAME%%.*}"  # Remove extension
CONFIG=$(dirname "${BASH_SOURCE[0]}")/../../config.sh
source $CONFIG

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
train_librispeechmix_none.py \
hparams/LibriSpeechMix/conformer-t_none.yaml \
--data_folder $DATA_DIR/LibriSpeechMix \
--splits [train-2mix_21Aug2023, dev-clean-2mix, test-clean-2mix] \
--test_splits [test-clean-2mix] \
--output_folder results/${SCRIPT_NAME} \
--num_epochs $NUM_EPOCHS \
--augment $AUGMENT \
--trim_nontarget 0.0 \
--num_targets min \
--suppress_delay True \
--gain_nontarget -60 \
--distributed_launch

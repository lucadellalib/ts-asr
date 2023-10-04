# NOTE: activate virtual environment (if any) before running this script

CONFIG=$(dirname "${BASH_SOURCE[0]}")/../config.sh
source $CONFIG

cd $ROOT_DIR
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
train_librispeechmix_pretrained.py \
hparams/LibriSpeechMix/conformer-t_wavlm.yaml \
--data_folder $DATA_DIR/LibriSpeechMix-21Aug2023 \
--output_folder results/wavlm/2mix_21Aug2023_WavLM_Overlap20_SpkEmbCat \
--num_epochs $NUM_EPOCHS \
--augment $AUGMENT \
--overlap_ratio 0.2 \
--injection_mode cat \
--distributed_launch

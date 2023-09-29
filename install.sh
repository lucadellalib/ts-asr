# Activate virtual environment: conda activate pytorch_p38

ROOT_DIR=workspace

mkdir $ROOT_DIR
cp -r /efs-storage/TS-ASR $ROOT_DIR/
cd $ROOT_DIR/TS-ASR
pip install -e vendor/speechbrain
pip install -r requirements.txt

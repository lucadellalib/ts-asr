# Activate virtual environment: conda activate pytorch_p38

mkdir workspace
cp -r /efs-storage/TS-ASR workspace/
cd workspace/TS-ASR
pip install -e vendor/speechbrain
pip install -r requirements.txt

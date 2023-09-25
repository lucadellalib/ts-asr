# Activate virtual environment: conda activate pytorch_p38

mkdir workspace24Sep
cp -r /efs-storage/TS-ASR workspace24Sep/
cd workspace24Sep/TS-ASR
pip install -e vendor/speechbrain
pip install -r requirements.txt

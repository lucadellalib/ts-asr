mkdir workspace
cp -r /efs-storage/TS-ASR workspace/
cd workspace/TS-ASR
conda activate pytorch_p38
pip install -e vendor/speechbrain
pip install -r requirements.txt

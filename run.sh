#!/bin/bash
#SBATCH --job-name=multitaskfon
#SBATCH --gres=gpu:rtx8000:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/multitask_fon/slurmerror.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/multitask_fon/slurmoutput.txt


###########cluster information above this line
module load python/3.9 cuda/10.2/cudnn/7.6
source /home/mila/b/bonaventure.dossou/env/bin/activate
cd /home/mila/b/bonaventure.dossou/multitask_fon
pip install -r requirements.txt
cd code
python run_train.py
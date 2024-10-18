#!/bin/bash
#SBATCH --account=rrg-bovy
#SBATCH --gpus-per-node=1
#SBATCH --mem=54000M
#SBATCH --time=0-02:50
#SBATCH --mail-user=nolan.koblischke@mail.utoronto.ca
#SBATCH --mail-type=ALL
module load python/3.11
module load scipy-stack
source /home/nkob/scratch/july2024/bin/activate
python finetuningAPOGEE.py
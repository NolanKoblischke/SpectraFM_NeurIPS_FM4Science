#!/bin/bash
#SBATCH --account=rrg-bovy
#SBATCH --gpus-per-node=4
#SBATCH --mem=400000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=nolan.koblischke@mail.utoronto.ca
#SBATCH --mail-type=FAIL

module load python/3.11
module load scipy-stack
source /home/nkob/scratch/july2024/bin/activate

python trainingAPOGEEGPU.py
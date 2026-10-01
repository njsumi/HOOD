#!/bin/bash
#SBATCH -A rayyeh
#SBATCH -p a10
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH --job-name=hood-check-inference
#SBATCH --output=hood-check-inference-%j.out

module reset
module load external conda/2025.09
source activate hood
export PYTHONNOUSERSITE=1
export HOOD_PROJECT=$HOME/HOOD
export HOOD_DATA=$RCAC_SCRATCH/hood_data

cd $HOOD_PROJECT
python scripts/check_inference.py



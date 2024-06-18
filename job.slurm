#!/bin/bash

#SBATCH --job-name=python-training
#SBATCH --output=slurm_out/result-%J.out
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:rtx6000:1
#
# Load Python and any other modules required for your job
# module load python/3.10

# Activate your virtual environment
source ./venv/bin/activate

# Execute the Python script with additional arguments
python src/trainer.py --batch_size 8

# Deactivate the virtual environment
deactivate

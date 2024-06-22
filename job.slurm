#!/bin/bash

#SBATCH --job-name=python-training
#SBATCH --output=slurm_out/result-%J.out
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:rtx6000:1

# Check if the config path argument is provided
if [ -z "$1" ]; then
    echo "Error: No config path provided. Usage: sbatch $0 <config_path>"
    exit 1
fi

# Load Python and any other modules required for your job
# module load python/3.10

# Activate your virtual environment
source ./venv/bin/activate

# Execute the Python script with the config path provided from the command line
python src/trainer.py --config_path="$1"

# Deactivate the virtual environment
deactivate


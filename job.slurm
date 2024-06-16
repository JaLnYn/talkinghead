#!/bin/bash

#SBATCH --job-name=python-training
#SBATCH --output=result-%J.out
#SBATCH --error=error-%J.err
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# Load Python and any other modules required for your job
# module load python/3.10

# Activate your virtual environment
source ./venv/bin/activate

# Navigate to the directory containing your script
# cd /path/to/your/project

# Execute the Python script with additional arguments
python src/trainer.py --batch_size 16

# Deactivate the virtual environment
deactivate

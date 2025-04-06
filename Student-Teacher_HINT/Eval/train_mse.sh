#!/bin/bash
#SBATCH --partition=comino
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

###############################################################################
# Usage:
#   sbatch --job-name="hint_${1}_${2}_${3}" \
#          --output="hint_${1}_${2}_${3}_%j.out" \
#          --error="hint_${1}_${2}_${3}_%j.err" \
#          train_mse.sh <base_name> <lambda> <d>
###############################################################################

# Grab command-line arguments
base_name=$1
_lambda=$2
d=$3

# Initialize conda (assumes Miniconda is installed in ~/miniconda3)
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your HINT environment
conda activate HINT_env
echo "Active conda environment: $(conda info --envs | grep '*' )"

# Confirm arguments
echo "Starting train_mse.py with arguments: base_name=$base_name, lambda=$_lambda, d=$d"

# Run your training script
python train_mse.py "$base_name" "$_lambda" "$d"

echo "Job complete."

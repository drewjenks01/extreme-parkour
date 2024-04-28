#!/bin/bash

#SBATCH --job-name=ep_parkour_
#SBATCH --partition=vision-pulkitag-3090
#SBATCH --qos=vision-pulkitag-main
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --time=48:00:00

source $H/exports.sh  # this exports the following: LD_LIBRARY_PATH=/data/pulkitag/misc/cbczhang/miniconda3/envs/llm-curriculum/lib/

# Set the Python script and its arguments
python_script="train_ep.py"

python_args="--exptid phase3_ep_no_reindex_go1_new --headless --resume_path"
python $python_script $python_args
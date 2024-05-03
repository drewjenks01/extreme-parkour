#!/bin/bash

#SBATCH --job-name=ep_parkour
#SBATCH --partition=vision-pulkitag-3090
#SBATCH --qos=vision-pulkitag-main
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=improbablex004
#SBATCH -o ./slurm_output/slurm-%j.out # STDOUT

source /data/scratch-oc40/pulkitag/awj/extreme-parkour/startup.rc
# Set the Python script and its arguments
python_script="train_ep.py"

# python_args="--exptid phase2_ep_no_reindex_go1_rgb --headless --resume_path /data/scratch-oc40/pulkitag/awj/extreme-parkour/legged_gym/logs/final_models/phase1_ep_no_reindex_go1/model_latest.pt --use_rgb"
python_args="--exptid phase1_go1_no_ang_vel --headless"
python -u $python_script $python_args > ./slurm_output/slurm-%j.out
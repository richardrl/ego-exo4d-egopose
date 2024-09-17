#!/bin/bash

#SBATCH --job-name=egopose_prep
#SBATCH --partition=vision-pulkitag-h100
#SBATCH --partition=vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-v100
#SBATCH --nodes=1
#SBATCH --qos=vision-pulkitag-main
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=60
#SBATCH --error=slurm_error-%A_%a.err

# activate mamba scripts
export CONDARC=/data/pulkitag/models/rli14/.condarc
source /data/pulkitag/models/rli14/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate

micromamba activate --prefix=/data/scratch-oc40/pulkitag/rli14/micromamba/ego4d_cli_scratchoc40

python3 main_v2.py --steps create_aria_calib extract_aria_img undistort_aria_img --ego4d_data_dir /data/pulkitag/models/rli14/data/egoexo --gt_output_dir /data/pulkitag/models/rli14/data/egoexo_hand_gen --portrait_view --split="val" 09102024_egoexoval_full --resolution=448 --mp
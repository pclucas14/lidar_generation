#!/bin/bash
#SBATCH --gres=gpu:1 # request GPU "generic resource"
#SBATCH --cpus-per-task=1 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=50000M # memory per
#SBATCH --time=0-3:00 # time (DD-HH:MM)
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rrg-dprecup

ssh gra-login2 -L 9000:127.0.01:443 -N -f
source ~/pytorch_py3/bin/activate
cd ~/lidar_generation/
python "$@"

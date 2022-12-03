#!/bin/bash
#SBATCH -o log/job.%j.out
#SBATCH -p GPU
#SBATCH --qos=normal
#SBATCH -J afm_go_9A_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:2

cd /gpfs/share/home/2000012508/structural_ml/afm_go-1
source /gpfs/share/home/2000012508/structural_ml/.afmvenv/bin/activate
python3 main.py --log-name 9A_all.log --mode train --worker 10

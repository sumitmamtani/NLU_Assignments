#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/home/sm9669/%j_%x.out
#SBATCH --error=/home/sm9669/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python ./run_hyperparameter_search.py
"

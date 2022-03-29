#!/bin/bash
#SBATCH --job-name=autoencoder
#SBATCH --error=autoencoder-%j.err
#SBATCH --output=autoencoder-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=longrun
#SBATCH --mem=40G

module load anaconda/3
module load cuda/11.4
module load cudnn/8.2


# conda run -n tf python train_autoencoder.py $SLURM_JOBID
# conda run -n tf python test_autoencoder.py directory-autoencoder/
conda run -n tf python train_anomaly_detector.py directory-autoencoder/ $SLURM_JOBID

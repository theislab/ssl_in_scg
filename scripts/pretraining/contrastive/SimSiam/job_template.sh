#!/bin/bash

#SBATCH -J SimSiam_{index}
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH --mem=80GB
#SBATCH --nice=10000

if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.bashrc"
fi

cd $SSL_PROJECT_HOME/self_supervision/trainer/contrastive/

python -u train.py --p {p} --negbin_intensity {negbin_intensity} --dropout_intensity {dropout_intensity} --max_epochs 12 --contrastive_method "SimSiam"

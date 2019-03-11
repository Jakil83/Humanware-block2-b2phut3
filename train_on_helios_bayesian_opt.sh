#!/bin/bash
#PBS -A colosse-users
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

export ROOT_DIR=$HOME'/branch_hblk2/Humanware-block2-b2phut3'
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'
export DATA_DIR='$SVHN_DIR/train $SVHN_DIR/extra'
export METADATA_FILENAME='$SVHN_DIR/train_metadata.pkl $SVHN_DIR/extra_metadata.pkl'

s_exec python $ROOT_DIR'/bayesian-opt.py' --dataset_dir $DATA_DIR --metadata_filename $METADATA_FILENAME --results_dir $ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml

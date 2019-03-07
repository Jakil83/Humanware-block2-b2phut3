#!/bin/bash
#PBS -A colosse-users
#PBS -l nodes=1:gpus=1
#PBS -l advres=MILA2019
#PBS -l feature=k80
#PBS -l walltime=00:30:00

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

export ROOT_DIR=$HOME'/Humanware-block2-b2phut3/'
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'
export DATA_DIR=[$SVHN_DIR/train, $SVHN_DIR/extra]
export METADATA_FILENAME=[$SVHN_DIR/train_metadata.pkl, $SVHN_DIR/extra_metadata.pkl]

s_exec python $ROOT_DIR'/train.py' --dataset_dir=$TMP_DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml
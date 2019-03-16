#!/bin/bash
#PBS -A colosse-users
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

# PROJECT_PATH will be changed to the master branch of your repo
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2phut3/code/'

RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2phut3/'
DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/test_sample'
METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/test_sample_metadata.pkl'
cd $PROJECT_PATH/evaluation

s_exec python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME

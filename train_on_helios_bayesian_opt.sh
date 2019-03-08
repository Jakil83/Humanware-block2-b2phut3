#!/bin/bash
#PBS -A colosse-users
#PBS -l nodes=1:gpus=1
#PBS -l walltime=012:00:00
# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

export ROOT_DIR=$HOME'/blk2_humanware/'
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'
export DATA_DIR=$SVHN_DIR/train
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$ROOT_DIR/tmp_results
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl'

mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR

if [ ! -f $SVHN_DIR'/train.tar.gz' ]; then
    
    echo "Downloading files for the training set!"
    wget -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

if [ ! -d $TMP_DATA_DIR ]; then

    echo "Extracting Files to " $TMP_DATA_DIR
    cp $DATA_DIR'/train.tar.gz' $TMP_DATA_DIR
    tar -xzf $TMP_DATA_DIR'/train.tar.gz' -C $TMP_DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

s_exec python $ROOT_DIR'/bayesian-opt.py' --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml

# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR 

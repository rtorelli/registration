#!/bin/bash -x

DATASET=${1:-ryan}
BUFFER_SIZE=8
N_ITER=100000
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}/${DATASET}.csv"
#PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/RT_RT_DAPI_Lamin/train.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

#python scripts/python/sample3.py ${DATASET} "data/csvs" \
#       --owner Forrest \
#       --project M247514_Rorb_1 \
#       --source REG_MARCH_21_DAPI_2 \
#       --target REG_MARCH_21_DAPI_1 \
#       --sample_length 1024 \
#       --samples_slice 10

#python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" \
#       --train_size 8.0 \
#       --seed 61 \
#       -v 

python train_model.py \
       --nn_module fnet_nn_2d \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --class_dataset ImageRegDataset \
       --transform_signal fnet.transforms.normalize \
       --transform_target fnet.transforms.normalize \
       --patch_size 256 256 \
       --batch_size 20 \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 16000 \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --iter_checkpoint 10000 20000 30000 40000 50000 60000 70000 80000 90000 

#!/bin/bash -x

DATASET=${1:-ryan}
MODEL_DIR="saved_models/DAPI2-3_DAPI1"
N_IMAGES=90
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

TEST_OR_TRAIN=test
python predict.py \
       --class_dataset ImageRegDataset \
       --transform_signal fnet.transforms.normalize \
       --transform_target fnet.transforms.normalize \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv "data/csvs/DAPI_2_safe.csv" \
       --n_images ${N_IMAGES} \
       --no_prediction_unpropped \
       --path_save_dir "/nas5/ryan/${DATASET}/${TEST_OR_TRAIN}" \
       --gpu_ids ${GPU_IDS} \
       --no_signal

#TEST_OR_TRAIN=train
#python predict.py \
#       --class_dataset ImageRegDataset \
#       --transform_signal fnet.transforms.normalize \
#       --transform_target fnet.transforms.normalize \
#       --path_model_dir ${MODEL_DIR} \
#       --path_dataset_csv "/pipeline/ryan/${TEST_OR_TRAIN}.csv"\
#       --n_images ${N_IMAGES} \
#       --no_prediction_unpropped \
#       --path_save_dir "results/${DATASET}/${TEST_OR_TRAIN}" \
#       --gpu_ids ${GPU_IDS}

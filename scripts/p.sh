#!/bin/bash -x

DATASET=${1:-ryan}
MODEL_DIR="saved_models/RT_DAPI_Lamin"
N_IMAGES=10
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

TEST_OR_TRAIN=test
python predict_ryan.py \
       --class_dataset ImageRegDataset \
       --transform_signal fnet.transforms.normalize \
       --transform_target fnet.transforms.normalize \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv "data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv" \
       --n_images ${N_IMAGES} \
       --no_prediction_unpropped \
       --path_save_dir "/nas5/ryan/${DATASET}/${TEST_OR_TRAIN}" \
       --gpu_ids ${GPU_IDS} 

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

#!/bin/bash -x
OWNER=Small_volumes_2018
PROJECT=M367240B_SSTPV_LaminB1_smallvol
INPUT_STACK=RT_STI_S01_DAPI_1
MODEL=RT_DAPI_Lamin
OUTPUT_STACK=RT_STI_S01_DAPI_1_smooth_enhance2
DIRECTORY="/nas5/ryan" 

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/create_smooth_avg.py \
       --dst_dir ${DIRECTORY} \
       --owner ${OWNER} \
       --project ${PROJECT} \
       --input_stack ${INPUT_STACK} \
       --output_stack ${OUTPUT_STACK} \
       --path_model_dir "saved_models/${MODEL}" \
       --module_fnet_model fnet_model \
       --gpu_ids 0

python scripts/python/import.py \
       --dst_dir ${DIRECTORY} \
       --owner ${OWNER} \
       --project ${PROJECT} \
       --client_scripts "/pipeline/render/render-ws-java-client/src/main/scripts" \
       --input_stack ${INPUT_STACK} \
       --output_stack ${OUTPUT_STACK} \
       --min_intensity 0 \
       --max_intensity 65535


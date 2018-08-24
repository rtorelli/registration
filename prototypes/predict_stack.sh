#!/bin/bash -x

OWNER=Small_volumes_2018
PROJECT=M367240_D_SSTPV_smallvol
INPUT_STACK=STI_FF_S03_DAPI_3
OUTPUT_STACK=LE_Test
MODEL=DAPI3_DAPI1 

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python prototypes/predict_stack.py ${OWNER} \
       ${PROJECT} \
       ${INPUT_STACK} \
       ${OUTPUT_STACK} \
       "saved_models/${MODEL}" \
       --dst_dir "/nas5/ryan"

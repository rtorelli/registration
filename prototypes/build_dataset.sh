#!/bin/bash -x

DATASET=leila
OWNER=Forrest
PROJECT=M247514_Rorb_1
SOURCE=REG_MARCH_21_DAPI_3
TARGET=REG_MARCH_21_DAPI_1

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python prototypes/build_dataset.py ${DATASET} \
    ${OWNER} \
    ${PROJECT} \
    ${SOURCE} \
    ${TARGET} \
    --sample_length 1024 \
    --samples_slice 10


docker run --rm -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
              -v $(cd "$(dirname ${BASH_SOURCE})"/.. && pwd):/root/projects/pytorch_fnet \
              -v /nas5:/nas5 \
              -v /pipeline:/pipeline \
              ${USER}/pytorch_fnet \
              bash


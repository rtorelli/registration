FROM pytorch/pytorch:v0.2

RUN apt update && apt install -y \
vim \
libx11-6 \
imagemagick \ 
 && rm -rf /var/lib/apt/lists/* 

# Jupyter Notebook config
COPY docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

COPY . /root/projects/pytorch_fnet
WORKDIR "/root/projects/pytorch_fnet"
RUN pip install -e . 

WORKDIR "/renderpython"
RUN git clone https://github.com/fcollman/render-python.git && \
        cd render-python && \ 
        python setup.py install

WORKDIR "/root/projects/pytorch_fnet"
ENV PYTHONUNBUFFERED 1

#### Install GEOS ####
# Inspired by: https://hub.docker.com/r/cactusbone/postgres-postgis-sfcgal/~/dockerfile/

ENV GEOS http://download.osgeo.org/geos/geos-3.5.0.tar.bz2

#TODO make PROCESSOR_COUNT dynamic
#built by docker.io, so reducing to 1. increase to match build server processor count as needed
ENV PROCESSOR_COUNT 1

WORKDIR /install-postgis

WORKDIR /install-postgis/geos
ADD $GEOS /install-postgis/geos.tar.bz2
RUN tar xf /install-postgis/geos.tar.bz2 -C /install-postgis/geos --strip-components=1
RUN ./configure && make -j $PROCESSOR_COUNT && make install
RUN ldconfig
WORKDIR /install-postgis
RUN pip install Shapely
WORKDIR "/root/projects/pytorch_fnet"

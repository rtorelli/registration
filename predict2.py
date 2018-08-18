import argparse
import fnet.data
import importlib
import json
import numpy as np
import os
import pandas as pd
import tifffile
import time
import torch
import warnings
import pdb
import renderapi
import requests
from PIL import Image 
import io

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result 

def save_tiff(tag, ar, path_tiff_dir):
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    tifffile.imsave(path_tiff, ar)
    print('saved:', path_tiff)
 
def main(): 
    host = 'http://10.128.24.33'
    port=80
    owner='Forrest'
    project='M247514_Rorb_1'
    stack='REG_MARCH_21_DAPI_3'    
    class_dataset='ImageRegDataset'
    gpu_ids=0
    module_fnet_model='fnet_model'
    path_model_dir='saved_models/3_1'
    path_save_dir='/pipeline/ryan/results'
     
    model = fnet.load_model(path_model_dir, gpu_ids, module=module_fnet_model)
    #zvalues = renderapi.stack.get_z_values_for_stack(stack, host=host, port=port, owner=owner, project=project)
    z=2
    path_z_dir = os.path.join(path_save_dir, str(int(z)))
    os.makedirs(path_z_dir)
    tilespecs = renderapi.tilespec.get_tile_specs_from_z(stack, z, 
                                                         host = host, port = port, 
                                                         owner = owner, project = project)
    for tile in tilespecs:
        request_url = renderapi.render.format_preamble(host, port, owner, project, stack) + \
                      "/tile/%s/tiff16-image" % (tile.tileId) + '?excludeMask=true'
        r = requests.Session().get(request_url)
        try:    
            img = Image.open(io.BytesIO(r.content))
            img = np.asarray(img)
        except:
            print('bad url')
            return
        #Slice to one channel and crop to 2048x2048
        img = img[:2048,:2048,0]
        img = normalize(img)
        img = torch.from_numpy(img).float() 
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        prediction = model.predict(img) if model is not None else None
        save_tiff('{:s}'.format(tile.tileId), prediction.numpy()[0, ], path_z_dir)
main()

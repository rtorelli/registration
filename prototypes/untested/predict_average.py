import fnet.data
import argparse
import tifffile
import torch
import requests
from PIL import Image 
import io
import os
import renderapi
import numpy as np
import pandas as pd
import json
import time

def normalize(img):
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('owner')
    parser.add_argument('project')
    parser.add_argument('input_stack')
    parser.add_argument('output_stack')
    parser.add_argument('path_model_dir')
    parser.add_argument('--dst_dir', default='/nas5/ryan')
    parser.add_argument('--host', default='http://10.128.24.33')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--client_scripts', default='/pipeline/render/render-ws-java-client/src/main/scripts')
    parser.add_argument('--module_fnet_model', default='fnet_model')
    parser.add_argument('--gpu_ids', type=int, default=0)
    opts = parser.parse_args()

    if not os.path.exists(opts.dst_dir):
        print('Destination directory does not exist')
        return
    path_project = os.path.join(opts.dst_dir, opts.project)
    if not os.path.exists(path_project):
        os.mkdir(path_project)
    path_output_stack = os.path.join(path_project, opts.output_stack)
    if os.path.exists(path_output_stack):
        print('Output stack exists.')
        return
    os.mkdir(path_output_stack) 
       
    render = renderapi.render.Render(host=opts.host, port=opts.port, 
                                                         owner=opts.owner, project=opts.project)
    #Load models
    initial = time.time()
    models = []
    for i in range(1,11):
        checkpoint_path = os.path.join(opts.path_model_dir, 'checkpoints')
        model_path = os.path.join(checkpoint_path, '{:06d}'.format(i * 10000))
        models.append(fnet.load_model(model_path, opts.gpu_ids,
                                      module=opts.module_fnet_model))
        print('load model ',i)
        if models[i-1] is None:
            print('bad model')
            exit(0)
    final = time.time()
    elapsed = final - initial
    print('load models elapsed time (sec): ', elapsed)

    #Determine min and max values of input stack
    initial = time.time() 
    stack_min = sys.maxsize
    stack_max = -sys.maxsize
    norm_stack_min = sys.maxsize
    norm_stack_max = -sys.maxsize
    zvalues = renderapi.stack.get_z_values_for_stack(opts.input_stack, render=render)
    for z in zvalues:
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.input_stack, z, render=render)
        for ts in tilespecs:
            #Request tile image
            request_url = renderapi.render.format_preamble(opts.host, opts.port, opts.owner, opts.project, opts.input_stack) + \
                          "/tile/%s/tiff-image?excludeMask=true&excludeAllTransforms=true" % (ts.tileId) 
            r = requests.Session().get(request_url)
            try:    
                img = Image.open(io.BytesIO(r.content))
                img = np.asarray(img)
            except:
                print('bad url for tile ', ts.tileId)
                continue
            img = img[:2048,:2048,0]
            tile_min = np.amin(img)
            tile_max = np.amax(img)
            if tile_min < stack_min: stack_min = tile_min
            if tile_max > stack_max: stack_max = tile_max
            #Normalize tile
            img = normalize(img)
            tile_min = np.amin(img)
            tile_max = np.amax(img)
            if tile_min < norm_stack_min: norm_stack_min = tile_min
            if tile_max > norm_stack_max: norm_stack_max = tile_max
    final = time.time()
    elapsed = final - initial
    print('stack min = ', stack_min)
    print('stack max = ', stack_max)
    print('normalized stack min = ', norm_stack_min)
    print('normalized stack max = ', norm_stack_max)
    print('scale factor elapsed time (sec): ', elapsed)
    
    #Predict tile images and specs
    initial = time.time() 
    tsarray = []
    for z in zvalues:
        for ts in tilespecs:
            #Request tile image
            request_url = renderapi.render.format_preamble(opts.host, opts.port, opts.owner, opts.project, opts.input_stack) + \
                          "/tile/%s/tiff-image?excludeMask=true&excludeAllTransforms=true" % (ts.tileId) 
            r = requests.Session().get(request_url)
            try:    
                img = Image.open(io.BytesIO(r.content))
                img = np.asarray(img)
            except:
                print('bad url for tile ', ts.tileId)
                continue
            #Crop, normalize, add two dimensions
            img = img[:2048,:2048,0]
            img = normalize(img)
            img = torch.from_numpy(img).float() 
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)       
            #Predict new image
            img_p = np.zeros((2048,2048), dtype=np.float32)
            for m in models:
                prediction = m.predict(img)
                img_p += prediction.numpy()[0,0,]
            img_p /= len(models)
            #Scale predicted image
            factor = (stack_max - stack_min) / (norm_stack_max - norm_stack_min)
            img_p *= factor
            img_p -= (factor * norm_stack_min)
            img_p += stack_min
            img_p = np.round(img_p)
            img_p = img_p.astype(np.uint16)
            #Save predicted tile image and new spec
            path_tiff = os.path.join(path_output_stack, '{:s}.tif'.format(ts.tileId))
            tifffile.imsave(path_tiff, img_p)
            d = ts.to_dict()
            d['mipmapLevels']['0']['imageUrl'] = 'file:' + path_tiff 
            ts.from_dict(d)
            tsarray.append(ts)
            print('saved:', path_tiff)
    final = time.time()
    elapsed = final - initial
    print('prediction elapsed time (sec): ', elapsed)

    #Create stack
    renderapi.stack.create_stack(opts.output_stack,
                                 stackResolutionX = 1,
                                 stackResolutionY = 1, 
                                 render=render)
    r = renderapi.render.connect(host=opts.host, 
                                 port=opts.port, 
                                 owner=opts.owner, 
                                 project=opts.project,
                                 client_scripts=opts.client_scripts)
    renderapi.client.import_tilespecs(opts.output_stack, tsarray, render=r)
    renderapi.stack.set_stack_state(opts.output_stack, 'COMPLETE',render=r)
            
    with open(os.path.join(path_output_stack, 'predict_options.json'), 'w') as outfile:
        json.dump(vars(opts), outfile, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()

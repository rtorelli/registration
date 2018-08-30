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
       
    render = renderapi.render.Render(host=opts.host, 
                                     port=opts.port, 
                                     owner=opts.owner, 
                                     project=opts.project)
    #Load model
    model = fnet.load_model(opts.path_model_dir, 
                            opts.gpu_ids, 
                            module=opts.module_fnet_model)
    if model is None:
        print('bad model')
        return

    #Create new tile images and specs 
    tsarray = []
    zvalues = renderapi.stack.get_z_values_for_stack(opts.input_stack, render=render)
    for z in zvalues:
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.input_stack, z, render=render)
        #TODO: break up 40-line for loop for readability
        for ts in tilespecs:
            #Request tile image
            request_url = renderapi.render.format_preamble(opts.host, 
                                                           opts.port, 
                                                           opts.owner, 
                                                           opts.project, 
                                                           opts.input_stack) + \
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
            prediction = model.predict(img)
            img_p = prediction.numpy()[0,0,]
            #Scale predicted image
            d = ts.to_dict()
            min_intensity = d['minIntensity'] 
            max_intensity = d['maxIntensity']
            img_min = -100.0
            img_max = 100.0
            factor = (max_intensity - min_intensity) / (img_max - img_min)
            img_p *= factor
            img_p -= (factor * img_min)
            img_p += min_intensity
            img_p = np.round(img_p)
            img_p = img_p.astype(np.uint16)
            #Save predicted tile image and new spec
            path_tiff = os.path.join(path_output_stack, '{:s}.tif'.format(ts.tileId))
            tifffile.imsave(path_tiff, img_p)
            d['mipmapLevels']['0']['imageUrl'] = 'file:' + path_tiff 
            ts.from_dict(d)
            tsarray.append(ts)
            print('saved:', path_tiff)

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

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

def normalize(img):
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst_dir', default='/nas5/ryan')
    parser.add_argument('--host', default='http://10.128.24.33')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--owner', default='Forrest')
    parser.add_argument('--project', default='M247514_Rorb_1')
    parser.add_argument('--input_stack', default='REG_MARCH_21_DAPI_2')
    parser.add_argument('--output_stack', default='RT2_REG_MARCH_21_DAPI_2')
    parser.add_argument('--path_model_dir', default='saved_models/2_1')
    parser.add_argument('--module_fnet_model', default='fnet_model')
    parser.add_argument('--gpu_ids', type=int, default=0)
    opts = parser.parse_args()

    path_images = os.path.join(opts.dst_dir, opts.output_stack)
    if not os.path.exists(path_images):
        os.mkdir(path_images)
    path_images_csv = os.path.join(path_images, 'images.csv')
    if os.path.exists(path_images_csv):
        print('Using existing stack.')
        return
       
    render = renderapi.render.Render(host=opts.host, 
                                     port=opts.port, 
                                     owner=opts.owner, 
                                     project=opts.project)
    #Load model
    #model = fnet.load_model(opts.path_model_dir, 
    #                        opts.gpu_ids, 
    #                        module=opts.module_fnet_model)
    #if model is None:
    #    print('bad model')
    #    exit(0)
    i = 10
    checkpoint_path = os.path.join(opts.path_model_dir, 'checkpoints')
    model_path = os.path.join(checkpoint_path, '{:06d}'.format(i * 10000))
    model = fnet.load_model(model_path,
                            opts.gpu_ids,
                            module=opts.module_fnet_model)
    if model is None:
        print('bad model')
        exit(0)

    #Create new tiles
    img_paths = []
    tsarray = []
    zvalues = renderapi.stack.get_z_values_for_stack(opts.input_stack, render=render)
    zvalues = [1008]
    #minpix=[]
    #maxpix=[]
    for z in zvalues:
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.input_stack, z, render=render)
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
                print('bad url')
                exit(0)
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
            img_min = -2
            img_max = 50
            factor = 65535 / (img_max - img_min)
            img_p *= factor
            img_p -= factor * img_min
            img_p = np.round(img_p)
            img_p = img_p.astype(np.uint16)
            #im = (img_p > 100) * img_p
            #minpix.append(np.amin(img_p))
            #maxpix.append(np.amax(img_p))
            #Save
            path_tiff = os.path.join(path_images, '{:s}.tif'.format(ts.tileId))
            tifffile.imsave(path_tiff, img_p)
            img_paths.append(path_tiff)
            print('saved:', path_tiff)
    #Write image paths to file
    print('image set size: ', len(img_paths))   
    df_img_paths = pd.DataFrame(data=np.array(img_paths), 
                                columns=['path'])
    df_img_paths.to_csv(path_images_csv, index=False)
    print('saved:', path_images_csv)                 
    #print('min',min(minpix))
    #print('max',max(maxpix))

if __name__ == '__main__':
    main()

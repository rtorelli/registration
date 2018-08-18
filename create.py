#if __name__ == "__main__" and __package__ is None:
#   __package__ = "renderapps.stack.create_clean_stack"
#
import fnet.data
import importlib
import tifffile
import torch
import requests
from PIL import Image 
import io
#
import json
import os
import renderapi
#from ..module.render_module import RenderModule,RenderParameters
#from json_module import InputFile,InputDir,OutputDir
#import marshmallow as mm
from functools import partial
import glob
import time
import numpy as np
#from functools import reduce
import operator

example_parameters={
    'render':{
        'host' : 'http://10.128.24.33',
        'port' : 80,
        'owner' : 'Antibody_testing_2018',
        'project' : 'M367240_B_SSTPV_Lamin_B1',
        'client_scripts' : '/pipeline/render/render-ws-java-client/src/main/scripts'
    },
    'input_stack' : 'RA00_STI_S01_DAPI_1_R5',
    'output_stack' : 'RT2_RA00_STI_S01_DAPI_1_R5',
    'output_directory' : '/nas5/ryan/dapi_lamin2',
    'gpu_ids' : 0,
    'module_fnet_model' : 'fnet_model',
    'path_model_dir' : 'saved_models/dapi_lamin2'
}

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

#def process_z():

class CreateCleanStack(object):
    def __init__(self,schema_type=None,*args,**kwargs):
        #if schema_type is None:
         #   schema_type = CreateCleanStackParameters
        #super(CreateCleanStack,self).__init__(schema_type=schema_type,*args,**kwargs)
        self.args = schema_type

    def run(self):
        if not os.path.exists(self.args['output_directory']):
            os.mkdir(self.args['output_directory'])
        
        render = renderapi.render.Render(host=self.args['render']['host'], 
                                         port=self.args['render']['port'], 
                                         owner=self.args['render']['owner'], 
                                         project=self.args['render']['project'])
        #Load model
        model = fnet.load_model(self.args['path_model_dir'], 
                                self.args['gpu_ids'], 
                                module=self.args['module_fnet_model'])
        if model is None:
            print('bad model')
            exit(0)
        
        #Create new tiles
        tsarray = []
        zvalues = renderapi.stack.get_z_values_for_stack(self.args['input_stack'], render=render)
        #zvalues = [86]
        #minpix=[]
        #maxpix=[]
        for z in zvalues:
            tilespecs = renderapi.tilespec.get_tile_specs_from_z(self.args['input_stack'], z, render=render)
            #tilespecs = renderapi.tilespec.get_tile_specs_from_z('REG_MARCH_21_DAPI_1', z, render=render)
            path_z_dir = os.path.join(self.args['output_directory'], str(int(z)))
            os.makedirs(path_z_dir)
            for ts in tilespecs:
                request_url = renderapi.render.format_preamble(self.args['render']['host'], 
                                                               self.args['render']['port'], 
                                                               self.args['render']['owner'], 
                                                               self.args['render']['project'], 
                                                               self.args['input_stack']) + \
                              "/tile/%s/tiff-image?excludeMask=true&excludeAllTransforms=true" % (ts.tileId) 
                r = requests.Session().get(request_url)
                try:    
                    img = Image.open(io.BytesIO(r.content))
                    img = np.asarray(img)
                except:
                    print('bad url')
                    exit(0)
                img = img[:2048,:2048,0]
                img = normalize(img)
                img = torch.from_numpy(img).float() 
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)

                #save prediction and print
                prediction = model.predict(img)
                img_p = prediction.numpy()[0,]
                #img_min = np.amin(img_p)
                #img_max = np.amax(img_p)
                img_min = -0.833853
                img_max = 49.6912
                factor = 255 / (img_max - img_min)
                img_p *= factor
                img_p -= factor * img_min
                img_p = np.round(img_p)
                #minpix.append(np.amin(img_p))
                #maxpix.append(np.amax(img_p))
                #pix=np.amin(prediction.numpy()[0,]) 
                #if pix < 0:
                #    im = prediction.numpy()[0,] - pix
                #    print('pix=',pix)
                #else:
                #    im = prediction.numpy()[0,]
                #im *= 100
                #im = np.uint32(im)
                #im=np.round(im)
                path_tiff = os.path.join(path_z_dir, '{:s}.tif'.format(ts.tileId))
                #im = prediction.numpy()[0,] * 5
                tifffile.imsave(path_tiff, img_p)
                
                print('saved:',path_tiff)
   
                #save signal and print
                #path_tiffb = os.path.join(path_z_dir, '{:s}b.tif'.format(ts.tileId))
                #tifffile.imsave(path_tiffb, img.numpy()[0,])
                #print(path_tiffb)
                
                #create tile
                d = ts.to_dict()
                d['mipmapLevels']['0']['imageUrl'] = 'file:' + path_tiff
                d['minIntensity'] = 0
                d['maxIntensity'] = 255
                ts.from_dict(d)
                tsarray.append(ts)
        #print('min',min(minpix))
        #print('max',max(maxpix))
        #print('minpix=',minpix)
        #print('maxpix=',maxpix)
        renderapi.stack.create_stack(self.args['output_stack'],
                                     stackResolutionX = 1,
                                     stackResolutionY = 1, 
                                     render=render)
        r = renderapi.render.connect(host=self.args['render']['host'], 
                                     port=self.args['render']['port'], 
                                     owner=self.args['render']['owner'], 
                                     project=self.args['render']['project'],
                                     client_scripts=self.args['render']['client_scripts'])
        renderapi.client.import_tilespecs(self.args['output_stack'], tsarray, render=r)
        renderapi.stack.set_stack_state(self.args['output_stack'], 'COMPLETE',render=r)

        #allzvalues = renderapi.stack.get_z_values_for_stack(self.args['input_stack'], render=render)
        #zvalues = np.array(allzvalues)	
        #mypartial = partial(process_z, render, self.args['render'], self.args['input_stack'], self.args['output_directory'], model)
        #tsarray.append(map(mypartial,zvalues))
        #with renderapi.client.WithPool(self.args['pool_size']) as pool:
        #    tsarray.append(pool.map(mypartial,zvalues))
        #renderapi.stack.create_stack(self.args['output_stack'], cycleNumber=6, 
        #                                            cycleStepNumber=1, stackResolutionX = 1, 
        #                                            stackResolutionY = 1, render=render)
        #renderapi.client.import_tilespecs_parallel(self.args['output_stack'],tsarray,render=render)
      
if __name__ == "__main__":
    mod = CreateCleanStack(schema_type=example_parameters)
    #mod = CreateCleanStack(schema_type=CreateCleanStackParameters)
    
    mod.run()

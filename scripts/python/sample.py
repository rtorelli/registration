import io
import sys
import renderapi
import numpy as np
import requests
from PIL import Image
import math
import random
import argparse
import os
import pandas as pd

#Finds box that inscribes mxn tilespec 
#Returns minX[M-1], minY[N-1], maxX[M-1], maxY[N-1]
def find_bounds(tilespecs, m, n):
    minX = []
    maxX = []
    maxY = []
    minY = []  
    for tile in tilespecs:
        corners = np.array([[0,0], [tile.width,0], 
                           [tile.width,tile.height], [0,tile.height]])
        transformed_corners = renderapi.transform.estimate_dstpts(tile.tforms, corners)
        minX.append(transformed_corners[0][0])
        maxX.append(transformed_corners[1][0])
        maxY.append(transformed_corners[2][1])
        minY.append(transformed_corners[0][1]) 
    minX.sort()
    maxX.sort(reverse=True)
    maxY.sort(reverse=True)
    minY.sort()
    return minX[m-1], maxX[m-1], maxY[n-1], minY[n-1]

#Finds parameters by which to translate maxX and minY due to a specific mask pattern
#Returns translation of width and height
def apply_mask(request_url):
    r = requests.Session().get(request_url)
    mask = Image.open(io.BytesIO(r.content))
    array = np.asarray(mask)
    height = 0
    width = 0 
    for x in reversed(range(array.shape[1])):
        if array[array.shape[0]-1][x][0] == 255:
            width = array.shape[1] - x - 1
            break
    for y in range(array.shape[0]):
        if array[y][0][0] == 255:
            height = y
            break
    return width, height

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset identifier')
    parser.add_argument('dst_dir', help='destination directory of dataset')
    parser.add_argument('--host', default='http://10.128.24.33')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--owner', default='Forrest')
    parser.add_argument('--project', default='M247514_Rorb_1')
    parser.add_argument('--source', default='REG_MARCH_21_DAPI_3')
    parser.add_argument('--target', default='REG_MARCH_21_DAPI_1')
    parser.add_argument('--m', type=int, default=6)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--sample_size', type=int, default=1024)
    parser.add_argument('--samples_slice', type=int, default=10)
    opts = parser.parse_args()

    path_dataset_csv = os.path.join(opts.dst_dir, opts.dataset + '.csv')
    if os.path.exists(path_dataset_csv):
        print('Using existing dataset csv.')
        return

    #Find common bounds for each slice
    render = renderapi.render.Render(opts.host, opts.port, opts.owner, opts.project)
    zvalues = renderapi.stack.get_z_values_for_stack(opts.source, render = render)
    samples = []
    for z in zvalues:
        #Find source bounds
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.source, z, render = render) 
        minXi,maxXi,maxYi,minYi = find_bounds(tilespecs, opts.m, opts.n) 
        #Find target bounds
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.target, z, render = render) 
        minXf,maxXf,maxYf,minYf = find_bounds(tilespecs, opts.m, opts.n) 
        #Translate bounds by mask
        request_url = renderapi.render.format_preamble(opts.host, opts.port, opts.owner, opts.project, opts.target) + \
                                                       '/tile/{}/mask/png-image'.format(tilespecs[0].tileId)
        w, h = apply_mask(request_url)
        #Find most restrictive bounds
        minX = max(minXi,minXf) 
        maxX = min(maxXi,maxXf) - w
        minY = max(minYi,minYf) + h 
        maxY = min(maxYi,maxYf)
        #Sample from slice
        for i in range(opts.samples_slice):
            x = random.randint(math.ceil(minX), math.floor(maxX) - opts.sample_size)
            y = random.randint(math.ceil(minY), math.floor(maxY) - opts.sample_size)
            samples.append ([opts.host, opts.port, opts.owner, opts.project, 
                             opts.source, opts.target, int(z), x, y, opts.sample_size])
    print('dataset size: ', len(samples))
    
    #Write sample coordinates to file
    df_samples = pd.DataFrame(data=np.array(samples), 
                              columns=['host', 'port', 'owner', 'project', 'source', 
                                       'target', 'z', 'x', 'y', 'size'])
    df_samples.to_csv(path_dataset_csv, index=False)
    print('saved:', path_dataset_csv)


if __name__ == '__main__':
    main()

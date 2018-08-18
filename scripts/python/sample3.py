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
from shapely import geometry

#Constant
WHITE = 255

#Finds polygonal shape of tilespec reduced by mask 
#Returns polygon
def find_bounds(tilespecs, min_x_edge, max_x_edge, min_y_edge, max_y_edge):
    tile = tilespecs[0]
    corners = np.array([[0,0], [tile.width,0], [tile.width,tile.height], [0,tile.height]])
    tcorners = renderapi.transform.estimate_dstpts(tile.tforms, corners)
    tcorners[0][0] += min_x_edge
    tcorners[3][0] += min_x_edge
    tcorners[1][0] -= max_x_edge
    tcorners[2][0] -= max_x_edge
    tcorners[0][1] += min_y_edge
    tcorners[1][1] += min_y_edge 
    tcorners[2][1] -= max_y_edge
    tcorners[3][1] -= max_y_edge   
    poly = geometry.Polygon(tcorners)
    iter_tilespecs = iter(tilespecs)
    next(iter_tilespecs)
    for tile in iter_tilespecs:
        corners = np.array([[0,0], [tile.width,0], [tile.width,tile.height], [0,tile.height]])
        tcorners = renderapi.transform.estimate_dstpts(tile.tforms, corners)
        tcorners[0][0] += min_x_edge
        tcorners[3][0] += min_x_edge
        tcorners[1][0] -= max_x_edge
        tcorners[2][0] -= max_x_edge
        tcorners[0][1] += min_y_edge
        tcorners[1][1] += min_y_edge 
        tcorners[2][1] -= max_y_edge
        tcorners[3][1] -= max_y_edge 
        poly = poly.union(geometry.Polygon(tcorners))
    return poly

#Finds parameters by which to crop tile due to mask
#Note: mask must not cover more than half tile from an edge
#      mask is non-white 
#Returns values for each tile edge
def find_mask_edges(request_url):
    r = requests.Session().get(request_url)
    if not r.status_code // 100 == 2:
        return 0, 0, 0, 0
    mask = Image.open(io.BytesIO(r.content))
    array = np.asarray(mask)
    min_x_edge = 0
    max_x_edge = 0
    min_y_edge = 0
    max_y_edge = 0
    for x in (range(array.shape[1])):
        if array[int(array.shape[0]/2)][x][0] == WHITE:
            min_x_edge = x
            break   
    for x in reversed(range(array.shape[1])):
        if array[int(array.shape[0]/2)][x][0] == WHITE:
            max_x_edge = (array.shape[1] - 1) - x
            break   
    for y in range(array.shape[0]):
        if array[y][int(array.shape[1]/2)][0] == WHITE:
            min_y_edge = y
            break  
    for y in reversed(range(array.shape[0])):
        if array[y][int(array.shape[1]/2)][0] == WHITE:
            max_y_edge = (array.shape[0] - 1) - y
            break
    return min_x_edge, max_x_edge, min_y_edge, max_y_edge 

#Finds minX and minY of sample with given length in poly 
#Returns x, y
def find_coord(poly, length):
    bb_bounds = poly.bounds
    minX = math.ceil(bb_bounds[0]) 
    maxX = math.floor(bb_bounds[2])
    minY = math.ceil(bb_bounds[1])
    maxY = math.floor(bb_bounds[3])
    x = 0
    y = 0
    is_within_poly = False
    while not is_within_poly:
        x = random.randint(minX, maxX - length)
        y = random.randint(minY, maxY - length)
        if poly.contains(geometry.Point([x,y])) and \
           poly.contains(geometry.Point([x+length,y])) and \
           poly.contains(geometry.Point([x+length,y+length])) and \
           poly.contains(geometry.Point([x,y+length])):
            is_within_poly = True
    return x, y

#Determines whether sample collides with other samples 
#Returns True or False
def is_collision(x, y, coords, length):
    for coord in coords:
        x_collision = False
        y_collision = False
        if (x > (coord[0] - length)) and (x < (coord[0] + length)):
            x_collision = True
        if (y > (coord[1] - length)) and (y < (coord[1] + length)):
            y_collision = True
        if (x_collision == True and y_collision == True):
            return True
    return False

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
    parser.add_argument('--sample_length', type=int, default=1024)
    parser.add_argument('--samples_slice', type=int, default=10)
    opts = parser.parse_args()

    path_dataset = os.path.join(opts.dst_dir, opts.dataset)
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)
    path_samples_csv = os.path.join(path_dataset, opts.dataset + '.csv')
    if os.path.exists(path_samples_csv):
        print('Using prior csv')
        return

    #Find common bounds for each slice
    render = renderapi.render.Render(opts.host, opts.port, opts.owner, opts.project)
    zvalues = renderapi.stack.get_z_values_for_stack(opts.source, render = render)
    samples = []
    for z in zvalues:
        #Find source bounds
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.source, z, render = render)
        request_url = renderapi.render.format_preamble(opts.host, opts.port, opts.owner, opts.project, opts.source) + \
                      '/tile/{}/mask/png-image'.format(tilespecs[0].tileId)
        min_x_edge, max_x_edge, min_y_edge, max_y_edge = find_mask_edges(request_url) 
        source_poly = find_bounds(tilespecs, min_x_edge, max_x_edge, min_y_edge, max_y_edge) 
        #Find target bounds
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.target, z, render = render)
        request_url = renderapi.render.format_preamble(opts.host, opts.port, opts.owner, opts.project, opts.target) + \
                      '/tile/{}/mask/png-image'.format(tilespecs[0].tileId)
        min_x_edge, max_x_edge, min_y_edge, max_y_edge = find_mask_edges(request_url) 
        target_poly = find_bounds(tilespecs, min_x_edge, max_x_edge, min_y_edge, max_y_edge) 
        #Find most restrictive bounds
        poly = source_poly.intersection(target_poly)
        #Sample from slice
        zsamples = []
        for i in range(opts.samples_slice):
            x,y = find_coord(poly, opts.sample_length)
            while is_collision(x, y, zsamples, opts.sample_length):
                x,y = find_coord(poly, opts.sample_length)
            zsamples.append((x,y))
            samples.append ([opts.host, opts.port, opts.owner, opts.project, 
                             opts.source, opts.target, int(z), x, y, opts.sample_length])
    #Write sample coordinates to file
    df_samples = pd.DataFrame(data=np.array(samples), 
                              columns=['host', 'port', 'owner', 'project', 'source', 
                                       'target', 'z', 'x', 'y', 'size'])
    df_samples.to_csv(path_samples_csv, index=False)
    print('sample set size: ', len(samples))
    print('saved:', path_samples_csv)

if __name__ == '__main__':
    main()

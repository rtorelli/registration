#import io
import argparse
import os
import renderapi
#import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst_dir', default='/nas5/ryan')
    parser.add_argument('--host', default='http://10.128.24.33')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--owner', default='Forrest')
    parser.add_argument('--project', default='M247514_Rorb_1')
    parser.add_argument('--client_scripts', default='/pipeline/render/render-ws-java-client/src/main/scripts')
    parser.add_argument('--input_stack', default='REG_MARCH_21_DAPI_2')
    parser.add_argument('--output_stack', default='RT2_REG_MARCH_21_DAPI_2')
    parser.add_argument('--min_intensity', type=int, default=0)
    parser.add_argument('--max_intensity', type=int, default=255)
    opts = parser.parse_args()

    #Read csv
    path_images = os.path.join(opts.dst_dir, opts.project, opts.output_stack)
    path_images_csv = os.path.join(path_images, 'images.csv')
    if not os.path.exists(path_images_csv):
        print('no csv')
        return
    df_img_paths = pd.read_csv(path_images_csv)

    #Import tiles
    render = renderapi.render.Render(host=opts.host, 
                                     port=opts.port, 
                                     owner=opts.owner, 
                                     project=opts.project)
    tsarray = []
    index = -1
    zvalues = renderapi.stack.get_z_values_for_stack(opts.input_stack, render=render)
    zvalues = [1008]
    for z in zvalues:
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(opts.input_stack, z, render=render)
        for ts in tilespecs:
            index += 1
            path = df_img_paths.iloc[index, 0]
            d = ts.to_dict()
            d['mipmapLevels']['0']['imageUrl'] = 'file:' + path
            d['minIntensity'] = opts.min_intensity
            d['maxIntensity'] = opts.max_intensity
            ts.from_dict(d)
            tsarray.append(ts)

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

if __name__ == '__main__':
    main()

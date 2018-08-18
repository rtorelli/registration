import renderapi
import os

host='http://10.128.24.33'
port=80
owner='Forrest'
project='M247514_Rorb_1'
client_scripts='/pipeline/render/render-ws-java-client/src/main/scripts'
stack='REG_MARCH_21_DAPI_3'
output_stack='AAB_REG_MARCH_21_DAPI_3'
output_dir='/nas5/ryan/DAPI_3'

'''Images for z=? have been generated'''
'''New stack has been created'''
'''Problem: Import tilespecs'''

zvalues = [17]
newts = []
for z in zvalues:
    tilespecs = renderapi.tilespec.get_tile_specs_from_z(stack, z, host=host,port=port,owner=owner,project=project)
    path_z_dir = os.path.join(output_dir, str(int(z)))
    for ts in tilespecs:
        path_tiff = os.path.join(path_z_dir, '{:s}.tif'.format(ts.tileId))
        d = ts.to_dict()
        d['mipmapLevels']['0']['imageUrl'] = 'file:' + path_tiff
        d['minIntensity'] = -5000
        d['maxIntensity'] = 5000
        ts.from_dict(d)
        newts.append(ts)
     
renderapi.stack.create_stack(output_stack, stackResolutionX = 1, stackResolutionY = 1,
                             host=host,port=port,owner=owner,project=project)
r = renderapi.render.connect(host=host,port=port,owner=owner,project=project,client_scripts=client_scripts)
renderapi.client.import_tilespecs(output_stack, newts, render=r)
renderapi.stack.set_stack_state(output_stack, 'COMPLETE',render=r)


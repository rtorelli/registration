import renderapi
import numpy as np
import pandas as pd
import pdb

class RenderAdaptor(object):
    def __init__(self, path_csv, img_format = 'tiff16', 
                 dataframe: pd.DataFrame = None):
        self.img_format = img_format
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['host','port','owner',
                                                  'project','source','target',
                                                  'z','x','y','size'])

    #Renders sample of source and target at index in df of sample coordinates 
    #Returns source and target as numpy arrays
    def get_item(self, index):
        row = self.df.iloc[index, :]
        host = row[0]
        port = row[1]
        owner = row[2]
        project = row[3]
        source = row[4]
        target = row[5]
        z = int(row[6])
        x = int(row[7])
        y = int(row[8])
        size = int(row[9])
        src = renderapi.image.get_bb_image(source, z, x, y, size, size, 
                                           img_format = self.img_format,
                                           host = host, port = port, 
                                           owner = owner, project = project)
        tgt = renderapi.image.get_bb_image(target, z, x, y, size, size, 
                                           img_format = self.img_format,
                                           host = host, port = port, 
                                           owner = owner, project = project) 
        return [src, tgt]

    #Returns length of df of sample coordinates
    def get_len(self): 
        return len(self.df)

    #Returns string of df at index
    def get_info(self, index):
        return self.df.iloc[index, 3:].to_dict()

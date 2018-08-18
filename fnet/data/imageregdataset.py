import torch
from fnet.data.renderadaptor import RenderAdaptor
from fnet.data.fnetdataset import FnetDataset
import fnet.transforms as transforms


class ImageRegDataset(torch.utils.data.Dataset):

    def __init__(self, path_csv: str = None, 
                 transform_source = [transforms.normalize],
                 transform_target = None):        
        self.transform_source = transform_source
        self.transform_target = transform_target

        self.adaptor = RenderAdaptor(path_csv)

    def __getitem__(self, index):
        im_out = self.adaptor.get_item(index)

        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])
        if self.transform_target is not None and (len(im_out) > 1):
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])
       
        im_out = [torch.from_numpy(im).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        return im_out
    
    def __len__(self):
        return self.adaptor.get_len()

    def get_information(self, index):
        return self.adaptor.get_info(index)

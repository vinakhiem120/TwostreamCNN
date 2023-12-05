import os
import pathlib
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from utils import utils

class ASLDataset(Dataset):
    def __init__(self, 
                 data_path,
                 transform = None):
        self.transform = transform
        self.img_path = list(pathlib.Path(data_path).glob('*/*.jpg'))
        self.class_name,self.class_to_idx = utils.get_class(data_path)
        
    def load_image(self,
                   index):
        path = self.img_path[index]
        return Image.open(path)
    
    def __getitem__(self,
                    idx):
        img = self.load_image(idx)
        if idx+1 >= len(self.img_path):
            img2 = self.load_image(idx)
        else:
            img2 = self.load_image(idx+1)
        class_name = self.img_path[idx].parent.name
        class_name_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img),self.transform(img2),class_name_idx
        else: 
            return img,img2, class_name_idx
    
    def __len__(self):
        return len(self.img_path) 
        
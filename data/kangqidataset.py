import os
from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data 

from torchvision.transforms import ToTensor,Compose
import h5py

class KangQiDataSet(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0
        self.input_channel = args.input_channel
        self.images_hr, self.images_lr = self._scan()
        self.repeat = 1
        self.toTensor = ToTensor()
    def _scan(self):
        if self.train:
            dataset = h5py.File('/data/kangqi/JointDemosaicSR/dataset/own/JointT.h5','r')
            images_hr = dataset['label']
            images_lr = dataset['train']
        else:
            dataset = h5py.File('/data/kangqi/JointDemosaicSR/dataset/own/JointTVad.h5','r')
            images_hr = dataset['label']
            images_lr = dataset['train']
        return images_hr, images_lr

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx
    
    
    
    def __getitem__(self, idx):
        
        lr, hr = self.images_lr[idx], self.images_hr[idx]
        lr, hr = self.toTensor(lr), self.toTensor(hr)
        lr = lr[0:1] + lr[1:2]+ lr[2:3]
        filename = str(idx + 1)
        return lr, hr, filename, self.idx_scale

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

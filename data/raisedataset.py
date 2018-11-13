import os
from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data 

class RAISEDataSet(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        self.images_hr, self.images_lr = self._scan()
    
    def _set_filesystem(self, dir_data):
        self.ppath = dir_data + 
    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        

import os
from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data 

import h5py

class RAISEDataSet300(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0
        self.input_channel = args.input_channel
        self._set_filesystem(args.dir_data, self.train)
        self.images_hr, self.images_lr = self._scan()
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]
        
        def _create_hdf5(path, data):
            f = h5py.File(path,'w')
            dt = h5py.special_dtype(vlen=np.dtype('int8'))
            dset = f.create_dataset("data", dtype=dt, data=data, chunks=True)
            f.close()

        def _load_hdf():
            self.images_hr = h5py.File(self._name_hrhdf(), 'r')['data']
            self.images_lr = [
                h5py.File(self._name_lrhdf(s))['data'] for s in self.scale               
            ]

        if args.ext == 'img':
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('npy') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.dir_data, 'RAISE_bin300')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        elif args.ext.find('hdf') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a hdf5 file')
                _load_hdf()
            except:
                print('Preparing a hdf5 file')
                hdf5_path = os.path.join(self.dir_data, 'RAISE_hdf')
                if not os.path.isdir(hdf5_path):
                    os.mkdir(hdf5_path)
                
                list_hr, list_lr = self._scan()
                hr = np.array([misc.imread(f) for f in list_hr])
                _create_hdf5(self._name_hrhdf(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = np.array([misc.imread(f) for f in list_lr[si]])
                    _create_hdf5(self._name_lrhdf(s), lr_scale)
                    del lr_scale
                _load_hdf()


        else:
            print('Please define data type')
    
    


    def _name_hrbin(self):
        return os.path.join(
            self.dir_data,
            'RAISE_bin300',
            '{}_bin_HR300.npy'.format(self.split)
        )
    
    def _name_hrhdf(self):
        return os.path.join(
            self.dir_data,
            'RAISE_hdf',
            '{}_hdf_HR.hdf5'.format(self.split)
        )
    def _name_lrbin(self, scale):
        return os.path.join(
            self.dir_data,
            'RAISE_bin300',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )
    
    def _name_lrhdf(self, scale):
        return os.path.join(
            self.dir_data,
            'RAISE_hdf',
            '{}_hdf_LR_X{}.hdf5'.format(self.split, scale)
        )


    def _set_filesystem(self, dir_data, is_train):
        self.dir_data = dir_data
        if is_train:
            self.dir_hr = os.path.join(dir_data, 'RAISE_train_HR300')
            self.dir_lr = os.path.join(dir_data, 'RAISE_train_LR_mosaic300')
        else:
            self.dir_hr = os.path.join(dir_data, 'RAISE_test_HR')
            self.dir_lr = os.path.join(dir_data, 'RAISE_test_LR_mosaic')
        self.ext = '.TIF'
        
    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        Imageset = os.listdir(self.dir_hr)
        self.filesname = Imageset
        for i, fn in enumerate(Imageset):
                    list_hr.append(os.path.join(self.dir_hr, fn))
                    for si, s in enumerate(self.scale):
                        list_lr[si].append(os.path.join(self.dir_lr, 'X{}'.format(s), fn))
        return list_hr, list_lr

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
    
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

    def  _load_file(self, idx):
        idx =  self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        filename = hr
        if self.args.ext == 'img':
            lr = misc.imread(lr)
            hr = misc.imread(hr)
            
        elif self.args.ext.find('npy') >=0:
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = self.filesname[idx]
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=True
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr
    
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        if self.input_channel == 1:
            lr_tensor = lr_tensor[0:1] + lr_tensor[1:2] + lr_tensor[2:]
        return lr_tensor, hr_tensor, filename, self.idx_scale









        

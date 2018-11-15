import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()
    
    def tic(self):
        self.t0 = time.time()
    
    def toc(self):
        return time.time() - self.t0
    
    def hold(self):
        self.acc += self.toc()
    
    def release(self):
        ret = self.acc
        self.acc = 0

        return ret
    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.psnrlog = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = './record/' + args.save
        else:
            self.dir = './record/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.psnrlog = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch{}...'.format(len(self.log)))
        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'
        
        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)
        
        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        
        self.plot_psnr(epoch)
        torch.save(self.psnrlog, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainner.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
    
    def add_psnrlog(self, psnr):
        self.psnrlog = torch.cat([self.psnrlog, psnr])
    
    def write_log(self, psnr, refresh=False):
        print(psnr)
        self.log_file.write(psnr + '\n')
        if refresh:
            self.log_file.flush()
    
    def done(self):
        self.log_file.close()
    
    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'PSNR results on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.psnrlog[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)
    
    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_X{}_'.format(self.dir, filename, scale)
        postfix = ('DemosSR', 'MosaicLR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)
    

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range):
    shave = scale
    diff = (sr - hr).data.div(rgb_range)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size = args.lr_decay,
            gamma = args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones=list(map(lambda x: int(x), milestones))
        scheduler=lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

import sys
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from torchvision import torch

sys.path.append('../diffusion-posterior-sampling/')
from LIME import LIME
import torch
import torch.nn as nn

# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def get_mean(self, x):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        return mean

    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class L_color(nn.Module): 

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


@register_operator(name='darken')
class DarkenOperator(NonLinearOperator):
    def __init__(self, img_path, w=256, h=256, dtype=torch.float32, device='cuda', beta=0.6, sigma=0.3):
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.sigma = sigma
        self.illum = None

        pic = Image.open(img_path).convert('RGB')
        self.origin_img = np.array(pic.resize((w, h), resample=Image.LANCZOS)).astype(np.float32) / 255.0
        self.lime = LIME(img=self.origin_img)
        # self.beta_map = nn.Parameter(torch.full((3, 256, 256), 0.75).to(self.device), requires_grad = True)
        self.beta_map = torch.full((3, 256, 256), 0.75).to(self.device)
        self.loss_exp = L_exp(32, 0.5)
        self.loss_color = L_color()

    def get_illum(self, step=1):
        with torch.no_grad():
            T_numpy = self.lime.run(step)
            T_ = torch.from_numpy(T_numpy).to(self.device, self.dtype)
            self.illum = T_.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            return self.illum

    def calculate_beta(self):
        delta = 1e-3
        delta = delta * torch.ones_like(self.illum)
        Lav = torch.mean(torch.log(self.illum + delta))
        maxL = torch.log(torch.min(self.illum) + delta)
        minL = torch.log(torch.max(self.illum) + delta)
        key = (Lav - minL) / maxL - minL
        # a_var = 3.0  = self.sigma
        self.beta = -(1 - self.sigma * (key - 0.6))
        print('beta: ', self.beta)
        return self.beta

    def forward(self, data, is_latent=False, adaptive_beta=False):
        if is_latent:
            data = (data + 1) / 2.0
        if adaptive_beta:
            meas = data * self.illum ** (torch.exp(self.illum) - self.sigma)
        else:
            meas = data * self.illum ** self.beta   
        if is_latent:
            meas = meas * 2.0 - 1.0
        return meas

    def transpose(self, meas, is_latent=False, adaptive_beta=False):
        if is_latent:
            meas = (meas + 1) / 2.0
        if adaptive_beta:
            # data = meas * self.illum ** (1-torch.exp(self.illum)-0.7)
            data = meas * self.illum ** (self.sigma - torch.exp(self.illum))
        else:
            data = meas * self.illum ** (-self.beta)
        if is_latent:
            data = data * 2.0 - 1.0
        return data

    # def forward(self, data, is_latent=False):
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     # meas = data * self.illum ** self.beta   
    #     meas = data * (self.illum ** self.cal_beta ) * self.alpha
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # # def transpose(self, meas, illum, is_latent=False):
    # def transpose(self, meas, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     # data = meas * self.illum ** (-self.beta)
    #     data = meas * (self.illum ** (-self.cal_beta)) / self.alpha
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data

    # def forward(self, data, illum, is_latent=False):
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     # meas = data * illum ** self.cal_beta * self.alpha
    #     meas = data * illum ** self.cal_beta 
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # def transpose(self, meas, illum, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     # data = meas * illum ** (-self.cal_beta) / self.alpha
    #     data = meas * illum ** (-self.cal_beta) 
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data

    # def forward(self, data, is_latent=False):
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     meas = data * self.illum ** self.beta_map 
    #     meas = data * self.illum ** self.beta_map 
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # def transpose(self, meas, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     data = meas * self.illum * (-1 ** self.beta_map) 
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data
    
    # def update_beta(self, data):
        
    #     beta_grad = torch.clone(self.beta_map.detach())
    #     beta_grad.requires_grad = True
        
    #     tmp = ((data + 1) / 2.0) * self.illum * (-1 * beta_grad)
    #     loss = self.loss_exp(tmp) + self.loss_color(tmp)
    #     loss.backward()
    #     # print(self.beta_map.grad)
    #     # with torch.no_grad():
    #     #     self.beta_map = self.beta_map - 1.0 * beta_grad.grad


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)

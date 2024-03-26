from dataclasses import dataclass
import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from lib.dataset.panostretch import uv_tri

class Compose:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomPanoStretch:
    def __call__(self, data):
        kx = np.random.rand()*1.5+0.5
        ky = np.random.rand()*1.5+0.5
        data = [self.panostretch(d.copy(), kx, ky) for d in data]
        return data

    def panostretch(self, data, kx, ky):
        sin_u, cos_u, tan_v = uv_tri(data.shape[1], data.shape[0])
        u0 = np.arctan2(sin_u * kx / ky, cos_u)
        v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

        refx = (u0 / (2 * np.pi) + 0.5) * data.shape[1] - 0.5
        refy = (v0 / np.pi + 0.5) * data.shape[0] - 0.5

        if (data.dtype == np.float32):
            data = np.stack([map_coordinates(data[..., i], [refy, refx], order=1, mode='wrap')
            for i in range(data.shape[-1])
            ], axis=-1)
        else:
            data = map_coordinates(data.astype('float32'), [refy, refx], order=0, mode='wrap')
            print(data)
            
        return data

class RandomGaussianNoise:
    def __call__(self, data):
        sigma = random.random()*20
        blur = sigma*np.random.randn(*data.shape)
        data = np.clip(data.copy() + blur, 0, 255).astype('uint8')
        return data

class RandomGaussianBlur:
    def __call__(self, data):
        sigma = random.random()*2
        tmp = gaussian_filter(data.copy(), sigma=(sigma, sigma, 0))
        return tmp



class RandomHorizontalShift:
    def __call__(self, data):
        width = data[0].shape[0]
        shift = np.floor((random.random()*(width))).astype('int')
        data = [np.roll(d, shift=shift, axis=1) for d in data]
        return data
        
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            # call copy() to avoid negative stride error in torch.from_numpy
            data = [d[:, ::-1].copy() for d in data]
        return data


class RandomCropandMirror:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            width = data[0].shape[0]
            crop = np.floor(random.random()*width/2).astype('int')
            data = [self.cropandmirror(d, crop) for d in data]
        return data

    def cropandmirror(self, data, crop):
        tmp = data.copy()
        width = tmp.shape[1]
        if (tmp.dtype == np.float32):
            tmp[:,0:int(width/2),:] = tmp[:,crop:int(width/2)+crop,:]
            tmp[:,int(width/2):width,:] = np.flip(tmp[:,0:int(width/2),:], axis=1)
        else:
            tmp[:,0:int(width/2)] = tmp[:,crop:int(width/2)+crop]
            tmp[:,int(width/2):width] = np.flip(tmp[:,0:int(width/2)], axis=1)
        return tmp






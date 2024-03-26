import os, glob
import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import cv2


def normalize_lab(lab_image):
    lab_image[:, :, 0] = lab_image[:, :, 0] / 50 - 1  # L: [0, 100] -> [-1, 1]
    lab_image[:, :, 1] = (lab_image[:, :, 1] ) / 128  # a: [-128, 127] -> [-1, 1]
    lab_image[:, :, 2] = (lab_image[:, :, 2] ) / 128  # b: [-128, 127] -> [-1, 1]
    return lab_image

def denormalize_lab(normalized_lab_image):
    normalized_lab_image[:, :, 0] = (normalized_lab_image[:, :, 0]+1) * 50  # L: [-1, 1] -> [0, 100]
    normalized_lab_image[:, :, 1] = (normalized_lab_image[:, :, 1] * 128)   # a: [-1, 1] -> [-128, 127]
    normalized_lab_image[:, :, 2] = (normalized_lab_image[:, :, 2] * 128)   # b: [-1, 1] -> [-128, 127]
    return normalized_lab_image


def convert_label(label, N_labels=50, shuffle=0):

    onehot = np.zeros((1, N_labels, label.shape[0], label.shape[1])).astype(np.float32)

    set_labels = np.unique(label)
    if (shuffle):
        set_labels = np.random.permutation(set_labels) #If real number of labels >> N_labels 

    ct = 0
    for t in set_labels.tolist():
        if ct >= N_labels:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class load_dataset:
    def __init__(self, root, split="train", color_transforms=None, geo_transforms=None):
        self.gt_dir = os.path.join(root, "groundTruth", split)
        self.img_dir = os.path.join(root, "images", split)

        self.index = os.listdir(self.gt_dir)

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms

        self.split = split


    def __getitem__(self, idx):
        idx = self.index[idx][:-4]
        gt = scipy.io.loadmat(os.path.join(self.gt_dir, idx+".mat"))
        t = np.random.randint(0, len(gt['groundTruth'][0]))
        if (gt['groundTruth'].shape[0]>1):
            gt = gt['groundTruth']
        else:
            gt = gt['groundTruth'][0][t][0][0][0]
        
        if os.path.exists(os.path.join(self.img_dir, idx+".png")):
            img = plt.imread(os.path.join(self.img_dir, idx+".png"))
            img = (img*255).astype('uint8')
        elif os.path.exists(os.path.join(self.img_dir, idx+".jpg")):
            img = plt.imread(os.path.join(self.img_dir, idx+".jpg")).astype('uint8')

        gt = cv2.resize(gt, dsize=(512,256), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, dsize=(512,256), interpolation=cv2.INTER_LINEAR)


        if self.color_transforms is not None:
            img = self.color_transforms(img)

        img = rgb2lab(img).astype(np.float32)
        img = normalize_lab(img)

        gt = gt.astype(np.int64)

        if self.geo_transforms is not None:
            img, gt = self.geo_transforms([img, gt])

        #Set N_labels according to the considered dataset 
        if (self.split == 'val'):
            N_labels = 800 
            gt = convert_label(gt, N_labels, shuffle=0)
            gt = torch.from_numpy(gt).reshape(N_labels, -1).float()
        else:
            N_labels = 200
            gt = convert_label(gt, N_labels, shuffle=1)
            gt = torch.from_numpy(gt).reshape(N_labels, -1).float()
            
        
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img, gt


    def __len__(self):
        return len(self.index)



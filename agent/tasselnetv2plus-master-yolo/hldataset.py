
import re
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd 
import random
import numpy as np
from PIL import Image
import cv2
import h5py
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

"NSPH label"
"""
value_dict = {
    1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 2, 11: 1, 12: 1, 13: 2,
    14: 2, 15: 0, 16: 1, 17: 2, 18: 2, 19: 0, 20: 0, 21: 1, 22: 0, 23: 2, 24: 0, 25: 1,
    26: 2, 27: 0, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 2, 34: 0, 35: 2, 36: 2, 37: 0,
    38: 0, 39: 0, 40: 2, 41: 2, 42: 2, 43: 1, 44: 1, 45: 2, 46: 0, 47: 1, 48: 2, 49: 0,
    50: 0, 51: 1, 52: 2, 53: 1, 54: 2
}
"""

"Variety label:0-D, 1-T"

value_dict = {
    1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 1, 12: 1, 13: 0,
    14: 1, 15: 0, 16: 0, 17: 1, 18: 1, 19: 0, 20: 1, 21: 0, 22: 0, 23: 1, 24: 1, 25: 0,
    26: 0, 27: 0, 28: 1, 29: 1, 30: 1, 31: 1, 32: 0, 33: 0, 34: 1, 35: 1, 36: 1, 37: 0,
    38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 1, 44: 0, 45: 1, 46: 1, 47: 1, 48: 0, 49: 1,
    50: 1, 51: 0, 52: 1, 53: 0, 54: 1
}


"Density label：H-0, M-1, L-2"
"""
value_dict = {
    1: 2, 2: 1, 3: 1, 4: 2, 5: 1, 6: 0, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 0, 13: 0,
    14: 0, 15: 0, 16: 0, 17: 1, 18: 2, 19: 2, 20: 2, 21: 0, 22: 1, 23: 0, 24: 1, 25: 2,
    26: 2, 27: 0, 28: 0, 29: 1, 30: 2, 31: 0, 32: 1, 33: 0, 34: 0, 35: 2, 36: 1, 37: 2,
    38: 1, 39: 0, 40: 1, 41: 1, 42: 2, 43: 1, 44: 2, 45: 1, 46: 2, 47: 2, 48: 0, 49: 0,
    50: 1, 51: 0, 52: 2, 53: 1, 54: 0
}
"""

def get_value_from_path(path, value_dict):
  
    match = re.search(r'\\(\d+)\.', path)
    if match:

        key = int(match.group(1))

        return value_dict.get(key)

    return None

def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, target, gtcount, label = sample['image'], sample['target'], sample['gtcount'], sample['label']
        h, w = image.shape[:2]

        if isinstance(self.output_size, tuple):
            new_h = min(self.output_size[0], h)
            new_w = min(self.output_size[1], w)
            assert (new_h, new_w) == self.output_size
        else:
            crop_size = min(self.output_size, h, w)
            assert crop_size == self.output_size
            new_h = new_w = crop_size
        if gtcount > 0:
            mask = target > 0
            ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[ch:h-ch+1, cw:w-cw+1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            if len(idh) != 0:
                ids = random.choice(range(len(idh)))
                hc, wc = idh[ids], idw[ids]
                top, left = hc-ch, wc-cw
            else:
                top = np.random.randint(0, h-new_h+1)
                left = np.random.randint(0, w-new_w+1)
        else:
            top = np.random.randint(0, h-new_h+1)
            left = np.random.randint(0, w-new_w+1)

        image = image[top:top+new_h, left:left+new_w, :]
        target = target[top:top+new_h, left:left+new_w]


        return {'image': image, 'target': target, 'gtcount': gtcount, 'label':label}


class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target, gtcount, label = sample['image'], sample['target'], sample['gtcount'], sample['label']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target, 'gtcount': gtcount, 'label':label}


class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target, gtcount, label = sample['image'], sample['target'], sample['gtcount'], sample['label']
        image, target = image.astype('float32'), target.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std

        image, target = image.astype('float32'), target.astype('float32')

        return {'image': image, 'target': target, 'gtcount': gtcount, 'label':label}


class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize =  self.psize
        image, target, gtcount, label = sample['image'], sample['target'], sample['gtcount'], sample['label']
        h,w = image.size()[-2:]
        ph,pw = (psize-h%psize),(psize-w%psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph!=psize) or (pw!=psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image,tmp_pad)
            target = F.pad(target,tmp_pad)

        return {'image': image, 'target': target, 'gtcount': gtcount, 'label':label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target, gtcount, label = sample['image'], sample['target'], sample['gtcount'], sample['label']
        image = image.transpose((2, 0, 1))
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        image, target = torch.from_numpy(image), torch.from_numpy(target)
        return {'image': image, 'target': target, 'gtcount': gtcount, 'label':label}


class MaizeTasselDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}


    def bbs2points(self, bbs):    
        points = []
        for bb in bbs:
            """
            x1, y1, w, h = [float(b) for b in bb]
            x2, y2 = x1+w-1, y1+h-1
            x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)
            """
            x, y = [float(b) for b in bb]
            x, y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
            points.append([x, y])
        return points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
            annotation = sio.loadmat(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if annotation['annotation'][0][0][1] is not None:
                bbs = annotation['annotation'][0][0][1]
                gtcount = bbs.shape[0]
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    cv2.circle(dotimage, (pt[0], pt[1]), int(24 * self.ratio) , (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, 80 * self.ratio)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})


        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class WhearEarDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
        self.labels = {}

    def parsexml(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()
        bbs = []
        for bb in root.iter('bndbox'):
            xmin = int(bb.find('xmin').text)
            ymin = int(bb.find('ymin').text)
            xmax = int(bb.find('xmax').text)
            ymax = int(bb.find('ymax').text)
            bbs.append([xmin, ymin, xmax, ymax])
        return bbs

    def bbs2points(self, bbs):    
        points = []
        for bb in bbs:
            x1, y1, x2, y2 = [float(b) for b in bb]
            x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)
            points.append([x, y])
        return points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
            label = get_value_from_path(file_name[0], value_dict)
            bbs = self.parsexml(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if bbs is not None:
                gtcount = len(bbs)
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    cv2.circle(dotimage, (pt[0], pt[1]), 6, (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, 40 * self.ratio)

            # cmap = plt.cm.get_cmap('jet')
            # target_show = target / (target.max() + 1e-12)
            # target_show = cmap(target_show) * 255.
            # target_show = 0.5 * image + 0.5 * target_show[:, :, 0:3]
            # plt.imshow(target_show.astype(np.uint8))
            # plt.show()
            # print(target.sum())

            # plt.imshow(dotimage.astype(np.uint8))
            # plt.show()

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})
            self.labels.update({file_name[0]: label})
        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]],
            'label': self.labels[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SorghumHeadDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
            annotations = read_image(self.data_dir+file_name[1])
            annotations = util.img_as_ubyte(annotations[:, :, 0]) == 0
            annotations = label(annotations, connectivity=annotations.ndim)
            annotations = regionprops(annotations)

            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if annotations is not None:
                gtcount = len(annotations)
                for annotation in annotations:
                    pt = annotation.centroid
                    x, y = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[x, y] = 1
                    cv2.circle(dotimage, (y, x), 6, (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, 8 * self.ratio)

            # cmap = plt.cm.get_cmap('jet')
            # target_show = target / (target.max() + 1e-12)
            # target_show = cmap(target_show) * 255.
            # target_show = 0.5 * image + 0.5 * target_show[:, :, 0:3]
            # plt.imshow(target_show.astype(np.uint8))
            # plt.show()
            # print(target.sum())

            # plt.imshow(dotimage.astype(np.uint8))
            # plt.show()

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})

        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=='__main__':

    dataset = WhearEarDataset(
        data_dir='./data/wheat_ears_counting_dataset', 
        data_list='./data/wheat_ears_counting_dataset/train.txt',
        ratio=0.167, 
        train=True, 
        transform=transforms.Compose([
            ToTensor()]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(len(dataloader))
    mean = 0.
    std = 0.
    for i, data in enumerate(dataloader, 0):
        images, targets = data['image'], data['target']
        bs = images.size(0)
        images = images.view(bs, images.size(1), -1).float()
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        print(images.size())
        print(i) 
    mean /= len(dataloader)
    std /= len(dataloader)
    print(mean/255.)
    print(std/255.)
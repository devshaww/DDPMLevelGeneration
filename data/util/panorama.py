from data.util.mask import (left_half_cropping_bbox, right_half_cropping_bbox, bbox2mask)
from torchvision import transforms
import numpy as np
import torch
import core.util as util
import os
import random


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # numpy: H x W x C
        # torch: C x H x W
        # map to [0, 1]
        ret = (sample.astype(float) / (len(util.REV_LOOKUP_TABLE) - 1)).transpose((2, 0, 1))  # HWC->CHW
        return torch.from_numpy(ret).float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # map from [0, 1] to [-1, 1]
        ret = (sample.add(-self.mean[0])).mul(1 / self.std[0])
        return ret


"""
input: (data(ndarray or tensor), filename)
is_origin: determine if it needs mask, no if set to True
is_left: determine if it needs left uncrop mask or right uncrop mask and tranforms

return: (16, 16) tensor with left or right half or both(when is_origin is True) filled with input
"""

# if is_origin, input[0] is ndarray
# if !is_origin, input[0] is tensor ranging from [-1,1]
# Many errors due to this method's flaw!!!!
def gen_starting_point(input, is_origin=False, is_left=True):
    filename = input[1]
    data = input[0]
    if torch.is_tensor(data):
        data = torch.squeeze(data).numpy()
    tfs = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    ndarray = np.zeros((16, 16), dtype=data.dtype)
    result = {}
    mask = get_mask(left=is_left, no_mask=is_origin)
    if is_origin:
        ndarray = data
    else:
        if is_left:
            ndarray[:, 8:] = data[:, 0:8]
        else:
            ndarray[:, 0:8] = data[:, 8:]

    if is_origin:
        img = ndarray.reshape(16, 16, 1)
        img = tfs(img)
    else:
        img = ndarray.reshape(1, 16, 16)
        img = torch.from_numpy(img)
    result['gt_image'] = img[None]
    result['cond_image'] = (img * (1. - mask) + mask * torch.randn_like(img))[None]
    result['mask_image'] = (img * (1. - mask) + mask)[None]
    result['mask'] = mask[None]
    result['path'] = [filename]

    return result


def get_mask(left=True, no_mask=False):
    if no_mask:
        mask = np.zeros((16, 16, 1), dtype=np.uint8)
    elif left:
        mask = bbox2mask((16, 16), left_half_cropping_bbox())
    else:
        mask = bbox2mask((16, 16), right_half_cropping_bbox())
    return torch.from_numpy(mask).permute(2, 0, 1)


# return (ndarray16x16:[0,27], filename)
def gen_rand_input():
    path = r"datasets/scenes/train"
    filelist = os.listdir(path)
    idx = random.randint(0, len(filelist)-1)
    file = filelist[idx]
    filepath = os.path.join(path, file)

    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    ndarray = np.zeros((16, 16), dtype=np.uint8)
    lis = []
    for st in lines:
        # return a matching number ranging from 0 to 27
        lis.append(list(map(util.lookup, st)))
    for i, val in enumerate(lis):
        ndarray[i, :] = val
    ndarray = ndarray.reshape(16, 16, 1)
    return ndarray, file


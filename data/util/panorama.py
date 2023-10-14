from data.util.mask import (left_half_cropping_bbox, right_half_cropping_bbox, bbox2mask)
from torchvision import transforms
import numpy as np
import torch
import core.util as util
import os
from data import dataset


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
input: (data, filename)
is_origin: determine if it needs mask and transforms, no if set to True
is_left: determine if it needs left uncrop mask or right uncrop mask

return: (16, 16) tensor with left or right half or both(when is_origin is True) filled with input
"""
# if is_origin, input[0] is ndarray
# if !is_origin, input[0] is tensor ranging from [-1,1]
def gen_starting_point(input, is_origin=False, is_left=True):
    filename = input[1]
    data = input[0]   # is_origin: data (16,16)  !is_origin: data (16,16,1)
    if torch.is_tensor(data):
        data = torch.squeeze(data).numpy()
    else:
        data = np.squeeze(data)
    size = data.shape  # (H, W)
    tfs = transforms.Compose([
        dataset.ToTensor(),
        dataset.Normalize(mean=[0.5], std=[0.5])
    ])
    ndarray = np.zeros(size, dtype=data.dtype)
    result = {}
    mask = get_mask(left=is_left, no_mask=is_origin)
    
    half_w = size[1] // 2
    if is_origin:
        img = data.reshape(size[0], size[1], 1)  # HWC
        img = tfs(img)
    else:
        if is_left:
            ndarray[:, half_w:] = data[:, 0:half_w]
            ndarray[:, :half_w] = -1.0
        else:
            ndarray[:, 0:half_w] = data[:, half_w:]
            ndarray[:, half_w:] = -1.0
        img = ndarray.reshape(1, size[0], size[1])  # CHW
        img = torch.from_numpy(img)

    result['gt_image'] = img[None]
    result['cond_image'] = (img * (1. - mask) + mask * torch.randn_like(img))[None]
    result['mask_image'] = (img * (1. - mask) + mask)[None]
    result['mask'] = mask[None]
    result['path'] = [filename.replace(".txt", "")]

    return result


def get_mask(left=True, no_mask=False, image_size=(16, 16)):
    if no_mask:
        mask = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)
    elif left:
        mask = bbox2mask(image_size, left_half_cropping_bbox())
    else:
        mask = bbox2mask(image_size, right_half_cropping_bbox())
    
    return torch.from_numpy(mask).permute(2, 0, 1)


# return (ndarray16x16:[0,27], filename)
def gen_rand_input_old(image_size=(16, 16)):
    path = r"datasets/scenes/train"
    filelist = os.listdir(path)
    idx = np.random.randint(0, len(filelist)-1)
    file = filelist[idx]
    filepath = os.path.join(path, file)

    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    ndarray = np.zeros(image_size, dtype=np.uint8)
    lis = []
    for st in lines:
        # return a matching number ranging from 0 to 27
        lis.append(list(map(util.lookup, st)))
    for i, val in enumerate(lis):
        ndarray[i, :] = val
    # ndarray = ndarray.reshape(16, 16, 1)
    ndarray = ndarray.reshape(image_size[0], image_size[1], 1)

    return ndarray, file


def gen_rand_input(image_size=(16, 16)):
    level = dataset.RANDOM_LEVEL
    if level is not None:
        filename = str(level['data_id'])
        if level['width'] <= image_size[0]:
            start_x = 0
        else:
            start_x = np.random.randint(level['width']-image_size[0])
        if level['height'] <= image_size[1]:
            start_y = 0
        else:
            start_y = level['height'] - image_size[1]

        level_objs = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
        for obj in level['objects']:
            obj_x = obj.x // 160
            obj_y = level['height'] - obj.y // 160 - 1
            if start_x + image_size[0] > obj_x >= start_x and start_y + image_size[1] > obj_y >= start_y:
                level_objs[obj_y - start_y, start_x - obj_x] = obj.id.value + 1

        for g in level['ground']:
            g_y = level['height'] - g.y - 1
            if start_x + image_size[0] > g.x >= start_x and start_y + image_size[1] > g_y >= start_y:
                level_objs[g_y - start_y, start_x - g.x] = 8

        level_objs = level_objs.reshape(image_size[0], image_size[1], 1)   # HWC

    else:
        raise NotImplementedError('RANDOM_LEVEL is None! Please make sure you have set it before panorama generation!')

    return level_objs, filename

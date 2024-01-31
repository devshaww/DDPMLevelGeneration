from data.util.mask import (left_half_cropping_bbox, right_half_cropping_bbox, bbox2mask)
import core.util as util
from data import dataset
import numpy as np
import torch
import torch.nn.functional as F
import pdb

"""
input: (data, filename)
is_origin: determine if it needs mask and transforms, no if set to True
is_left: determine if it needs left uncrop mask or right uncrop mask

return: (16, 16) tensor with left or right half or both(when is_origin is True) filled with input
"""
# if is_origin, input[0] is ndarray
# if !is_origin, input[0] is tensor ranging from [-1,1]
def gen_starting_point_old(input, is_origin=False, is_left=True):
    filename = input[1]
    data = input[0]   # is_origin: data (c,16,16)  !is_origin: data (16,16,c)
    if torch.is_tensor(data):
        data = torch.squeeze(data).numpy()
    else:
        data = np.squeeze(data)
    size = data.shape  # (H, W)
    tfs = transforms.Compose([
        dataset.ToTensor()
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


# input: is_origin ? (cx16x16 float tensor, filename) : (1xcx16x16 float tensor, filename)
# return: dict ["": (1xcx16x16)tensor, ...]
def gen_starting_point(input, is_origin=False, is_left=True):
    data, filenames = input[0], input[1]
    mask = get_mask(left=is_left, no_mask=is_origin)

    img = torch.zeros_like(data, dtype=data.dtype) 
    if not is_origin:   
        if is_left:
            img[:, :, :, 8:] = data[:, :, :, 0:8]
            # fill the unmasked half with background
            img[:, 0, :, :8] = 1.0
        else:
            img[:, :, :, 0:8] = data[:, :, :, 8:]
            # fill the unmasked half with background
            img[:, 0, :, 8:] = 1.0
    else:
        img = data

    result = {}

    img = F.softmax(img, dim=1) if is_origin else (img * (1. - mask) + mask * F.softmax(img, dim=1))
    cond_image = img * (1. - mask) + mask * torch.randn_like(img)
    cond_image = img * (1. - mask) + mask * F.softmax(cond_image, dim=1)

    result['gt_image'] = img
    result['cond_image'] = cond_image
    result['mask_image'] = (img * (1. - mask) + mask)
    result['mask'] = mask
    result['path'] = filenames

    return result


def get_mask(left=True, no_mask=False, image_size=(16, 16)):
    if no_mask:
        mask = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)
    elif left:
        mask = bbox2mask(image_size, left_half_cropping_bbox())
    else:
        mask = bbox2mask(image_size, right_half_cropping_bbox())
    
    return torch.from_numpy(mask).permute(2, 0, 1)


# return (nchw tensor, filename)
def gen_rand_input(paths, image_size=(16, 16)):
    container = []
    if paths is not None:
        for path in paths:
            with open(path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            container.append(lines)
        
        objs = np.zeros((len(paths), util.NUM_OF_OBJS, image_size[0], image_size[1]), dtype=np.uint8)
        for i,level in enumerate(container):
            for y in range(len(level)):
                for x in range(len(level[0])):
                    obj_id = util.MAIF_Encoding[level[y][x]]
                    # obj_id = util.MAIF_Encoding.get(lines[y][x], 0)
                    objs[i, obj_id, y, x] = 1
        
    else:
        raise NotImplementedError('Path is None! Please make sure you have set it before panorama generation!')

    return torch.from_numpy(objs).float(), [path.rsplit("/")[-1].replace(".txt", "") for path in paths]


def gen_rand_input_old(level, image_size=(16, 16)):
    if level is not None:
        filename = level['data_id']
        if level['width'] <= image_size[1]:
            start_x = 0
        else:
            start_x = np.random.randint(level['width']-image_size[1])
        start_y = 0
        # 2. crop
        level_objs = np.zeros((util.NUM_OF_OBJS, image_size[0], image_size[1]), dtype=np.uint8)
        level_objs[-1, :, :] = 1
        for obj in level['objects']:
            x, y = obj['x'], obj['y']
            if start_x + image_size[1] > x >= start_x and start_y + image_size[0] > y >= start_y:
                y = image_size[0] - y - 1
                level_objs[obj['id'], y+start_y:y+start_y+obj['h'], x-start_x : x-start_x+obj['w']] = 1
                level_objs[-1, y+start_y : y+start_y+obj['h'], x-start_x : x-start_x+obj['w']] = 0

        for g in level['ground']:
            if start_x + image_size[1] > g['x'] >= start_x and start_y + image_size[0] > g['y'] >= start_y:
                y = image_size[0] - g['y'] - 1
                level_objs[7, y+start_y, g['x']-start_x] = 1
                level_objs[-1, y+start_y, g['x']-start_x] = 0

    else:
        raise NotImplementedError('RANDOM_LEVEL is None! Please make sure you have set it before panorama generation!')

    # return level_objs, filename
    return torch.from_numpy(level_objs).float(), filename

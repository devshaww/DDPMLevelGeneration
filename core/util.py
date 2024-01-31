import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import os
import cv2
import json
import pdb
from PIL import Image

MARIO_CHARS = ['-', 'X', '#', 'S', 'C', 'L', 'U',
               '@', '?', '!', 'Q', '2', '1', 'D', 'o', 't', 'T',
               '*', '|', '%', 'g', 'E', 'G', 'r', 'R', 'k', 'K',
               'y', 'Y', 'B', 'b']

LOOKUP_TABLE = {'-': 0, '#': 1, 'S': 2, 'C': 3, 'L': 4, 'U': 5,
                '@': 6, '?': 6, '!': 7, 'Q': 7, '2': 8, '1': 9, 'D': 0, 'o': 10,
                't': 11, 'T': 12, '*': 13, '|': 0, '%': 2, 'g': 14, 'E': 14, 'G': 15,
                'r': 16, 'R': 17, 'k': 18, 'K': 19, 'y': 20, 'Y': 21, 'X': 22, 'B': 23, 'b': 24}

REV_LOOKUP_TABLE = {0: '-', 1: '#', 2: 'S', 3: 'C', 4: 'L', 5: 'U',
                    6: '@', 7: '!', 8: '2', 9: '1', 10: 'o',
                    11: 't', 12: 'T', 13: '*', 14: 'g', 15: 'G',
                    16: 'r', 17: 'R', 18: 'k', 19: 'K', 20: 'y', 21: 'Y', 22: 'X', 23: 'B', 24: 'b'}

NUM_OF_OBJS_backup = 13
NUM_OF_OBJS = 19

BG_COLOR_RGB = [99, 144, 255]

# (row, column)
# pipe: (single)13 15  bullet_bill_blaster: 19  semisolid_platform: 22
tile_pos_dict_backup = {0: (5, 2), 1: (0, 2), 2: (0, 6), 3: (1, 0), 4: (1, 7), 5: (6, 4), 6: (0, 3), 7: (5, 7),
                 8: (5, 3), 9: (0, 1), 13: (6, 4), 14: (6, 5), 15: (2, 2), 16: (2, 3), 17: (2, 4), 18: (2, 5),
                 19: (0, 3), 20: (0, 4), 21: (0, 5), 22: (5, 3), 23: (5, 4), 24: (5, 5), 25: (5, 6)}

# 4,5 invisible block   16开始全部+3 16-
tile_pos_dict = {0: (5, 2), 1: (0, 2), 2: (0, 6), 3: (0, 6), 4: (0, 6), 5: (0, 6), 6: (0, 6), 7: (0, 6), 8: (1, 0), 9: (1, 0), 10: (1, 7), 11: (6, 4), 
                 12: (0, 3), 13: (5, 7), 14: (5, 3), 15: (0, 1), 19: (6, 4), 20: (6, 5), 21: (2, 2), 22: (2, 3), 23: (2, 4), 24: (2, 5), 25: (0, 3),
                 26: (0, 4), 27: (0, 5), 28: (5, 3), 29: (5, 4), 30: (5, 5), 31: (5, 6)}

                #  13: (6, 4), 14: (6, 5), 15: (2, 2), 16: (2, 3), 17: (2, 4), 18: (2, 5),
                #  19: (0, 3), 20: (0, 4), 21: (0, 5), 22: (5, 3), 23: (5, 4), 24: (5, 5), 25: (5, 6)}

sprite_pos_dict_backup = {10: (5, 0), 11: (3, 0), 12: (7, 0)}

sprite_pos_dict = {16: (5, 0), 17: (3, 0), 18: (7, 0)}

# 13: single_pipe_head, 14: single_pipe_body, 15: pipe_head_left, 16: pipe_head_right, 17: pipe_body_left, 18: pipe_body_right 19: blaster_head
# 20: blaster_neck, 21: blaster_body, 22: single_semisolid_platform, 23: semisolid_platform_left, 24: semisolid_platform_right, 25: semisolid_platform_middle
extra_pos_dict_backup = {13: (6, 4), 14: (6, 5), 15: (2, 2), 16: (2, 3), 17: (2, 4), 18: (2, 5), 19: (0, 3), 
                         20: (0, 4), 21: (0, 5), 22: (5, 3), 23: (5, 4), 24: (5, 5), 25: (5, 6)}

# 19: single_pipe_head, 20: single_pipe_body, 21: pipe_head_left, 22: pipe_head_right, 23: pipe_body_left, 24: pipe_body_right 25: blaster_head
# 26: blaster_neck, 27: blaster_body, 28: single_semisolid_platform, 29: semisolid_platform_left, 30: semisolid_platform_right, 31: semisolid_platform_middle
extra_pos_dict = {19: (6, 4), 20: (6, 5), 21: (2, 2), 22: (2, 3), 23: (2, 4), 24: (2, 5), 25: (0, 3),
                 26: (0, 4), 27: (0, 5), 28: (5, 3), 29: (5, 4), 30: (5, 5), 31: (5, 6)}

MAIF_Encoding_backup = {
    'M': 0,   # mario
    'F': 0,   # exit
    '-': 0,   # background
    '#': 1,   # hard block
    'S': 2,   # block
    'U': 2,   # mushroom block
    '2': 2,   # invisible coin block
    '1': 2,   # invisible life-up block
    'C': 2,   # coin block
    'L': 2,   # life-up block
    '@': 3,   # mushroom question block
    '?': 3,   
    '!': 3,   # coin question block
    'Q': 3,   
    'D': 0,   # used block, should not appear in a normal level
    'o': 4,   # coin
    't': 5,   # pipe
    'T': 5,   # flower pipe
    '*': 6,   # bullet bill blaster
    'B': 6,   # bullet bill blaster head
    'b': 6,   # bullet bill blaster neck and body
    '|': 7,   # semisolid platform body
    '%': 8,   # semisolid platform
    'X': 9,   # ground
    'g': 10,  # goomba
    'E': 10,
    'G': 10,  # winged goomba
    'r': 11,  # red koopa
    'R': 11,  # winged red koopa
    'k': 11,  # green koopa
    'K': 11,  # winged green koopa
    'y': 12,  # spiky
    'Y': 12,  # winged spiky
}

LEVEL2STR_LOOKUP_MAP = {0: '-', 1: '#', 2: 'S', 3: 'U', 4: '2', 5: '1',
                    6: 'C', 7: 'L', 8: '@', 9: '!', 10: 'o',
                    11: 't', 12: '*', 13: '|', 14: '%', 15: 'X',
                    16: 'g', 17: 'r', 18: 'y'}

# Replace '?' to '@', 'Q' to '!', 'T' to 't', 'B'/'b' to '*', 'E' to 'g', 'R' to 'r', 'K' to 'k', 'Y' to 'y'

MAIF_Encoding = {
    'M': 0,   # mario
    'F': 0,   # exit
    '-': 0,   # background
    '#': 1,   # hard block
    'S': 2,   # block
    'U': 3,   # mushroom block
    '2': 4,   # invisible coin block
    '1': 5,   # invisible life-up block
    'C': 6,   # coin block
    'L': 7,   # life-up block
    '@': 8,   # mushroom question block
    '?': 8,   
    '!': 9,   # coin question block
    'Q': 9,   
    'D': 0,   # used block, should not appear in a normal level
    'o': 10,   # coin
    't': 11,   # pipe
    'T': 11,   # flower pipe
    '*': 12,   # bullet bill blaster
    'B': 12,   # bullet bill blaster head
    'b': 12,   # bullet bill blaster neck and body
    '|': 13,   # semisolid platform body
    '%': 14,   # semisolid platform
    'X': 15,   # ground
    'g': 16,  # goomba
    'E': 16,
    'G': 16,  # winged goomba
    'r': 17,  # red koopa
    'R': 17,  # winged red koopa
    'k': 17,  # green koopa
    'K': 17,  # winged green koopa
    'y': 18,  # spiky
    'Y': 18,  # winged spiky
}

def read_pos_dict():
    with open('data/spritesheet.json', 'r') as sf:
        sf_dic = json.load(sf)
    with open('data/tileset.json', 'r') as tf:
        tf_dic = json.load(tf)

    return sf_dic, tf_dic

sf_dic, tf_dic = read_pos_dict()


def get_frame(n):
    name = enum_reverse_mapping.get(n, 'background')
    if name in sf_dic:
        return sf_dic[name], False, name
    elif name in tf_dic:
        return tf_dic[name], True, name
    else:
        return None


def get_pos(n):
    if n in tile_pos_dict:
        return True, tile_pos_dict[n]
    elif n in sprite_pos_dict:
        return False, sprite_pos_dict[n]
    return None


def read_template():
    path_tileset = r"data/img/mapsheet.png"
    path_spritesheet = r"data/img/enemysheet.png"

    # tileset_img = cv2.imread(path_tileset, cv2.IMREAD_UNCHANGED)
    # spritesheet_img = cv2.imread(path_spritesheet, cv2.IMREAD_UNCHANGED)
    # tileset_img = np.asarray(cv2.cvtColor(cv2.imread(path_tileset), cv2.COLOR_BGR2RGB))  # HWC
    # spritesheet_img = np.asarray(cv2.cvtColor(cv2.imread(path_spritesheet), cv2.COLOR_BGR2RGB))
    
    tileset_img = np.array(Image.open(path_tileset))
    spritesheet_img = np.array(Image.open(path_spritesheet))
    
    tileset_img = np.transpose(tileset_img, (2, 0, 1))  # CHW
    spritesheet_img = np.transpose(spritesheet_img, (2, 0, 1))  # CHW

    return torch.from_numpy(tileset_img), torch.from_numpy(spritesheet_img)


# return a matching number ranging from 0 to 27
def lookup(x):
    return LOOKUP_TABLE[x]


# platform postprocess
def pf_postprocess(tensor):   # N C H W
	n, row, col = tensor.shape[0], tensor.shape[2], tensor.shape[3]
	for num in range(n):
		for i in range(row - 1):
			for j in range(col):
				if i >= row - 2:
					last2lines_list = ['-', 'X']  # '-': background   'X': ground
					choice = random.choices(last2lines_list, weights=[1, 19], k=1)[0]
					tensor[num, 0, i:, j] = LOOKUP_TABLE[choice]


def scene_postprocess(tensors):
	return [pf_postprocess(t) for t in tensors]


# tensor to txt and save
def tensor2txt(tensor):
    #  NCHW
	t = torch.squeeze(tensor, 1)  # n x 16 x 16
	t = (t.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
	t = (t * (len(REV_LOOKUP_TABLE) - 1)).type(torch.uint8)
	n, row, col = t.shape[0], t.shape[-2], t.shape[-1]
	for num in range(n):
		lines = []
		# path = os.path.join(dst_dir, f"epoch{epoch}_{num}.txt")  # ./results/TXT/epoch1_1.txt
		for i in range(row):
			line = ''
			for j in range(col):
				element = t[num, i, j].item()  # int
				line += REV_LOOKUP_TABLE[element]
			lines.append(line)
	return lines


def save_txt(lines, filename, dst_dir):
	path = os.path.join(dst_dir, f"{filename}.txt")
	with open(path, 'w') as f:
		f.writelines(line + '\n' for line in lines)

# handle pipe, bullet_bill_blaster, semisolid_platform
# pipe: (single)13 15  bullet_bill_blaster: 19  semisolid_platform: 22
def process_unfixed_element_backup(t):
    h, w = t.shape[-2], t.shape[-1]
    level_tiles = torch.zeros_like(t)
    for row in range(h):
        img_row = []  # container for 3x16x16 tensor
        row_data = t[row, :]  # (w, )
        for col in range(w):
            i = row_data[col].item()
            if i == 5:
                temp = 0
                singlePipe = False
                if col < w-1 and row_data[col+1].item() != 5 and col > 0 and row_data[col-1].item() != 5:
                    singlePipe = True
                if col > 0 and (level_tiles[row, col-1].item() == 15 or level_tiles[row, col-1].item() == 17):  # 15: pipe_head_left / 17: pipe_body_left
                    temp += 1
                if row > 0 and t[row-1, col].item() == 5:  # 上一行同一列为pipe元素
                    temp = temp + 1 if singlePipe else temp + 2
                level_tiles[row, col] = 13 + temp if singlePipe else 15 + temp
            elif i == 6:  # bullet_bill_blaster
                temp = 0
                if row > 0 and t[row-1, col].item() == 6: 
                    temp += 1
                if row > 1 and t[row-2, col].item() == 6:
                    temp += 1
                level_tiles[row, col] = 19 + temp 
            elif i == 8:  # semisolid platform
                temp = 0
                if col > 0 and t[row, col-1].item() == 8:
                    temp += 2
                if col < w - 1 and t[row, col+1].item() == 8:
                    temp += 1
                level_tiles[row, col] = 22 + temp
            else:
                level_tiles[row, col] = i

    return level_tiles


# handle pipe, bullet_bill_blaster, semisolid_platform
# pipe: (single)16 18  bullet_bill_blaster: 22  semisolid_platform: 25
def process_unfixed_element(t):
    h, w = t.shape[-2], t.shape[-1]
    level_tiles = torch.zeros_like(t)
    for row in range(h):
        img_row = []  # container for 3x16x16 tensor
        row_data = t[row, :]  # (w, )
        for col in range(w):
            i = row_data[col].item()
            if i == 11:  # pipe
                temp = 0
                singlePipe = False
                if col < w-1 and row_data[col+1].item() != 11 and col > 0 and row_data[col-1].item() != 11: #左右列都不为pipe元素
                    singlePipe = True
                if col > 0 and (level_tiles[row, col-1].item() == 21 or level_tiles[row, col-1].item() == 23): #同行左列为pipe元素
                    temp += 1
                if row > 0 and t[row-1, col].item() == 11: # 上一行同一列为pipe元素
                    temp = temp + 1 if singlePipe else temp + 2
                level_tiles[row, col] = 19 + temp if singlePipe else 21 + temp
            elif i == 12:  # bullet_bill_blaster
                temp = 0
                if row > 0 and t[row-1, col].item() == 12:
                    temp += 1
                if row > 1 and t[row-2, col].item() == 12:
                    temp += 1
                level_tiles[row, col] = 25 + temp
            elif i == 14:  # semisolid platform
                temp = 0
                if col > 0 and t[row, col-1].item() == 14:
                    temp += 2
                if col < w - 1 and t[row, col+1].item() == 14:
                    temp += 1
                level_tiles[row, col] = 28 + temp
            else:
                level_tiles[row, col] = i

    return level_tiles

# tensor: (c,h,w)
def better_visualize(tensor, tileset, spritesheet):
    shape = tensor.shape
    h, w = shape[-2], shape[-1]
    tile_h, tile_w = 16, 16
    # new_tensor = torch.round(tensor.mul(0.5).add(0.5).mul(NUM_OF_OBJS)).type(torch.uint8).squeeze()
    target = torch.ones((3, h * tile_h, w * tile_w), dtype=torch.uint8) * torch.tensor(BG_COLOR_RGB).reshape(3,1,1)  # RGB CHW

    bg_element = torch.ones((3, tile_h, tile_w), dtype=torch.uint8) * torch.tensor(BG_COLOR_RGB).reshape(3,1,1)
    t = torch.argmax(tensor, dim=0)  # (h, w)
    level_tiles = process_unfixed_element(t)

    for row in range(h):
        img_row = []  # container for 3x16x16 tensor
        row_data = level_tiles[row, :]  # (w, )
        for col in range(w):
            i = row_data[col].item()
            info = get_pos(i)   # have to use .item(), or it can not access dictionary by index, so it always gets default 'background', so the result is all empty
            element = bg_element
            if info is not None:
                is_tile, pos = info
                x0, y0 = pos[1] * tile_w, pos[0] * tile_h
                x1, y1 = x0 + tile_w, y0 + tile_h   # all elements have (1xtile_h, 1xtile_w) size
                if i == 17:  # koopa
                    y0 -= tile_h
                element = tileset[:, y0:y1, x0:x1] if is_tile else spritesheet[:, y0:y1, x0:x1]
                # handle alpha
                alpha_channel = element[3, :, :]
                alpha_mask = alpha_channel == 0
                if torch.any(alpha_mask):
                    # (3, #of transparent pixels)
                    element[:3, alpha_mask] = torch.tensor(BG_COLOR_RGB, dtype=torch.uint8).reshape(3,1)
                if i == 17 and row > 1:
                    target[:, (row-1)*tile_h : row*tile_h, col*tile_w : col*tile_w+tile_w] = element[:3, :16, :] 
            # img_row.append(element[:3, :, :])
            img_row.append(element[:3, -16:, :])

        target[:, row*tile_h : row*tile_h+tile_h, :] = torch.cat(img_row, dim=-1)  # (3,16,256)

    return target


def tensor2img(tensor, tileset, spritesheet, out_type=np.uint8, min_max=(-1, 1), nrow_=None):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()

    if n_dim == 4:
        shape = tensor.shape
        new_tensor = torch.zeros(shape[0], 3, shape[2]*16, shape[3]*16)  # NCHW
        n_img = len(tensor)  # N
        if not nrow_:
            nrow_ = int(math.sqrt(n_img))
        for i in range(n_img):
            new_tensor[i] = better_visualize(tensor[i], tileset, spritesheet)
        img_np = make_grid(new_tensor, nrow=nrow_, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = better_visualize(tensor, tileset, spritesheet).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = better_visualize(tensor, tileset, spritesheet).numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    return img_np.astype(out_type).squeeze()


def tensor2img2(tensor, out_type=np.uint8, min_max=(-1, 1), nrow_=None):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        if not nrow_:
            nrow_ = int(math.sqrt(n_img))
        img_np = make_grid(tensor, nrow=nrow_, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()    # (img_np + 1) / 2 * 255
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()


def postprocess(images, tileset, spritesheet):
	return [tensor2img(image, tileset, spritesheet) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args



### Just for Training Data Preparation and Data Augmentation.
### Will Never Get Run in Actual Training Process

# split a complete level to multiple 16x16 scene
# level should be a 16xn list
def split_level(level_path, prefix=''):
    with open(level_path, 'r') as f:
        lines = f.readlines()
        level = [line.strip() for line in lines]
    level_name = level_path[level_path.rfind('/') + 1:].replace('.txt', '')
    h, w = len(level), len(level[0])
    split_num, remainder = w // 16, w % 16
    for i in range(split_num):
        splits = []
        for j in range(h):
            split = level[j][i * 16:i * 16 + 16]
            splits.append(split)
        path = f"datasets/scenes/train/{prefix}_{level_name}_split{i}.txt"
        # path = os.path.join(os.path.abspath('..'), f"Palette/datasets/scenes/train/{prefix}_{level_name}_split{i}.txt")
        with open(path, 'w') as f:
            f.writelines(line + '\n' for line in splits)
    # remainder
    if remainder != 0:
        fill = ['-' * 16 for _ in range(16)]
        fill[-2] = 'X' * 16
        fill[-1] = 'X' * 16
        # start = split_num
        # end = split_num + remainder
        for j in range(16):
            line = level[j][split_num * 16:]  # remainder cols
            fill[j] = line + fill[j][remainder:]
        path = f"datasets/scenes/train/{prefix}_{level_name}_split{i+1}.txt"
        # path = os.path.join(os.path.abspath('..'), f"Palette/datasets/scenes/train/{level_name}_split{i + 1}.txt")
        with open(path, 'w') as f:
            f.writelines(line + '\n' for line in fill)


# exchange left half and right half of a 16x16 txt file to get a new txt file, a way of data augmentation
# data should be a 16x16 ndarray
def exchange_left_right(scene_path):
    with open(scene_path, 'r') as f:
        lines = f.readlines()
        scene = [line.strip() for line in lines]
    scene_name = scene_path[scene_path.rfind('/') + 1:].replace('.txt', '')
    h, w = len(scene), len(scene[0])
    splits = []
    for j in range(h):
        right = scene[j][w // 2:] if w % 2 == 0 else scene[j][w // 2 + 1]
        left = scene[j][0:w // 2]
        split = right + left
        splits.append(split)
    path = f"datasets/scenes/train/{scene_name}_rl.txt"
    with open(path, 'w') as f:
        f.writelines(line + '\n' for line in splits)

# abspath = os.path.abspath('..')
# split_level(os.path.join(abspath, 'datasets/levels/lvl-1.txt'))
# exchange_left_right(os.path.join(abspath, 'datasets/scenes/train/000000000001_0.txt'))

def gen_flist(dir):
    for fname in os.listdir(dir):
        with open('datasets/scenes/flist/train2.flist', 'a') as f:
            if fname.endswith('.txt'):
                path = os.path.join(dir, fname)
                f.write(path + '\n')


# if __name__ == '__main__':
#     # gen_flist('datasets/scenes/train')
#     # abspath = os.path.abspath('..')
#     # path = os.path.join(abspath, 'Palette/datasets/levels/original')
#     for root, dirs, files in os.walk('datasets/levels'):
#         prefix = root.split('/')[-1]
#         print(prefix)
#         for file in files:
#             path = os.path.join(root, file)
#             split_level(path, prefix)



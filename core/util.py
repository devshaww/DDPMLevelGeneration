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

NUM_OF_OBJS = 133

BG_COLOR_RGB = [99, 144, 255]

tile_pos_dict = {0: (5, 2), 1: (0, 2), 2: (0, 7), 3: (0, 7), 4: (0, 7), 5: (0, 7), 6: (1, 0), 7: (1, 0),
                 8: (1, 6), 9: (1, 6), 10: (1, 7), 11: (6, 4), 12: (6, 4), 22: (0, 1), 25: (0, 0)}

sprite_pos_dict = {13: (11, 0), 14: (5, 0), 15: (5, 0), 16: (3, 3), 17: (3, 3), 18: (3, 3), 19: (3, 3), 23: (11, 0),
                   24: (11, 0), 20: (7, 0), 21: (7, 0)}


# map id value to string name to find its position
enum_reverse_mapping = {
    0: "goomba",
    1: "koopa",
    2: "piranha_flower",
    3: "hammer_bro",
    4: "block",
    5: "question_block",
    6: "hard_block",
    7: "ground",
    8: "coin",
    9: "pipe",
    10: "spring",
    11: "lift",
    12: "thwomp",
    13: "bullet_bill_blaster",
    14: "mushroom_platform",
    15: "bob_omb",
    16: "semisolid_platform",
    17: "bridge",
    18: "p_switch",
    19: "pow",
    20: "super_mushroom",
    21: "donut_block",
    22: "cloud",
    23: "note_block",
    24: "fire_bar",
    25: "spiny",
    26: "goal_ground",
    27: "goal",
    28: "buzzy_beetle",
    29: "hidden_block",
    30: "lakitu",
    31: "lakitu_cloud",
    32: "banzai_bill",
    33: "one_up",  # mushroom get from block
    34: "fire_flower",
    35: "super_star",
    36: "lava_lift",
    37: "starting_brick",
    38: "starting_arrow",
    39: "magikoopa",
    40: "spike_top",
    41: "boo",
    42: "clown_car",
    43: "spikes",
    44: "big_mushroom",
    45: "shoe_goomba",
    46: "dry_bones",
    47: "cannon",
    48: "blooper",
    49: "castle_bridge",
    50: "jumping_machine",
    51: "skipsqueak",
    52: "wiggler",
    53: "fast_conveyor_belt",
    54: "burner",
    55: "door",
    56: "cheep_cheep",
    57: "muncher",
    58: "rocky_wrench",
    59: "track",
    60: "lava_bubble",
    61: "chain_chomp",
    62: "bowser",
    63: "ice_block",
    64: "vine",
    65: "stingby",
    66: "arrow",
    67: "one_way",
    68: "saw",
    69: "player",
    70: "big_coin",
    71: "half_collision_platform",
    72: "koopa_car",
    73: "cinobio",
    74: "spike_ball",
    75: "stone",
    76: "twister",
    77: "boom_boom",
    78: "pokey",
    79: "p_block",
    80: "sprint_platform",
    81: "smb2_mushroom",
    82: "donut",
    83: "skewer",
    84: "snake_block",
    85: "track_block",
    86: "charvaargh",
    87: "slight_slope",
    88: "steep_slope",
    89: "reel_camera",
    90: "checkpoint_flag",
    91: "seesaw",
    92: "red_coin",
    93: "clear_pipe",
    94: "conveyor_belt",
    95: "key",
    96: "ant_trooper",
    97: "warp_box",
    98: "bowser_jr",
    99: "on_off_block",
    100: "dotted_line_block",
    101: "water_marker",
    102: "monty_mole",
    103: "fish_bone",
    104: "angry_sun",
    105: "swinging_claw",
    106: "tree",
    107: "piranha_creeper",
    108: "blinking_block",
    109: "sound_effect",
    110: "spike_block",
    111: "mechakoopa",
    112: "crate",
    113: "mushroom_trampoline",
    114: "porkupuffer",
    115: "cinobic",
    116: "super_hammer",
    117: "bully",
    118: "icicle",
    119: "exclamation_block",
    120: "lemmy",
    121: "morton",
    122: "larry",
    123: "wendy",
    124: "iggy",
    125: "roy",
    126: "ludwig",
    127: "cannon_box",
    128: "propeller_box",
    129: "goomba_mask",
    130: "bullet_bill_mask",
    131: "red_pow_box",
    132: "on_off_trampoline",
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


def read_template():
    path_tileset = r"data/img/tileset.png"
    path_spritesheet = r"data/img/spritesheet.png"

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


def better_visualize(tensor, tileset, spritesheet):
    shape = tensor.shape
    h, w = shape[-2], shape[-1]
    tile_h, tile_w = 16, 16
    new_tensor = torch.round(tensor.mul(0.5).add(0.5).mul(NUM_OF_OBJS)).type(torch.uint8).squeeze()
    target = torch.ones((3, h * tile_h, w * tile_w), dtype=torch.uint8) * torch.tensor(BG_COLOR_RGB).reshape(3,1,1)  # RGB CHW

    for row in range(h):
        for col in range(w):
            # tstart_y: target image's starting point of y coordinate
            tstart_y = row * tile_h
            tstart_x = col * tile_w
            info = get_frame(new_tensor[row][col].item()-1)
            ts_alpha = tileset[3,:,:]
            ss_alpha = spritesheet[3,:,:]
            if info is not None:
                frame, is_tile, name = info
                # source start row(column), tileset and spritesheet's starting point of row(column)
                start_y = frame['y'] * tile_h
                start_x = frame['x'] * tile_w

                # source end row(column), according to the element's height(width) and the bounds
                if tstart_y + frame['height'] * tile_h > target.shape[1]:
                    end_y = start_y + (target.shape[1] - tstart_y)
                else:
                    end_y = start_y + frame['height'] * tile_h
                if tstart_x + frame['width'] * tile_w > target.shape[2]:
                    end_x = start_x + (target.shape[2] - tstart_x)
                else:
                    end_x = start_x + frame['width'] * tile_w

                # target end row(column), according to the element's height(width) and the bounds
                tend_y = min(target.shape[1], tstart_y + frame['height'] * tile_h)
                tend_x = min(target.shape[2], tstart_x + frame['width'] * tile_w)

                tile_tensor = tileset[:3, start_y:end_y, start_x:end_x] if is_tile else spritesheet[:3, start_y:end_y, start_x:end_x]
                target[:, tstart_y:tend_y, tstart_x:tend_x] = tile_tensor

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




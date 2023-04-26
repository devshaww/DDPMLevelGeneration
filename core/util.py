import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import os
import cv2
from PIL import Image
'''
'-':background   √
'X':ground  √
'#': paramidBlock   √
'S': normal block   √
'C': coin block (same as normal block but has coin)  √
'L': 1up block (same as normal block but has 1 extra life)   √
'U': mushroom block (same as normal block but has mushroom)  √
'@','?': mushroom question block  √
'!', 'Q': coin question block   √
'2': invisible coin block  √
'1': invisible 1 up block  √
'D': used  (use this tile for denoting invisible blocks) x  
'o': coin   √
't': empty pipe    √
'T': flower pipe   √
'*': bullet bill  √ 
'|': background for jump through block (green background, use background to replace)    x
'%': jump through block  (green muchroom block like a platform, use normal block to replace)  x
'g', 'E': GOOMBA   √
'G': GOOMBA_WINGED √
'r': RED_KOOPA  √
'R': RED_KOOPA_WINGED  √
'k': KOOPA  √
'K': KOOPA_WINGED  √
'y': SPIKY  √
'Y': SPIKY_WINGED  √
'B': bulet bill head  x
'b': bullet bill neck and body  x
'''
# not generate '|' '%' 'D' 'b' 'B' '*' elements
MARIO_CHARS = ['-', 'X', '#', 'S', 'C', 'L', 'U',
               '@', '?', '!', 'Q', '2', '1', 'D', 'o', 't', 'T',
               '*', '|', '%', 'g', 'E', 'G', 'r', 'R', 'k', 'K',
               'y', 'Y', 'B', 'b']

# update
LOOKUP_TABLE = {'-': 0, '#': 1, 'S': 2, 'C': 3, 'L': 4, 'U': 5,
                '@': 6, '?': 6, '!': 7, 'Q': 7, '2': 8, '1': 9, 'D': 0, 'o': 10,
                't': 11, 'T': 12, '*': 13, '|': 0, '%': 2, 'g': 14, 'E': 14, 'G': 15,
                'r': 16, 'R': 17, 'k': 18, 'K': 19, 'y': 20, 'Y': 21, 'X': 22, 'B': 23, 'b': 24}

REV_LOOKUP_TABLE = {0: '-', 1: '#', 2: 'S', 3: 'C', 4: 'L', 5: 'U',
                    6: '@', 7: '!', 8: '2', 9: '1', 10: 'o',
                    11: 't', 12: 'T', 13: '*', 14: 'g', 15: 'G',
                    16: 'r', 17: 'R', 18: 'k', 19: 'K', 20: 'y', 21: 'Y', 22: 'X', 23: 'B', 24: 'b'}
#update
'''
floor(X22): 0, 1
paramid_block(#1): 0, 2
block(%, S2, L4, U5, C3): 0, 7
invisible block(19, 28): 1, 6
pipe(T12 t11): 6, 4
question block(@6 ?6 !7 Q7): 1, 0
coin(o10): 1, 7
background(-0, |): 5, 2
mask: 25

spiky(y20 Y21):  7, 0
goomba(g14 E14 G15): 5, 0
koopa(r16 R17 k18 K19): 3, 3
bullet bill(*13 B23 b24): 11, 0
'''

# update
tile_pos_dict = {0: (5, 2), 1: (0, 2), 2: (0, 7), 3: (0, 7), 4: (0, 7), 5: (0, 7), 6: (1, 0), 7: (1, 0),
                 8: (1, 6), 9: (1, 6), 10: (1, 7), 11: (6, 4), 12: (6, 4), 22: (0, 1), 25: (0, 0)}

sprite_pos_dict = {13: (11, 0), 14: (5, 0), 15: (5, 0), 16: (3, 3), 17: (3, 3), 18: (3, 3), 19: (3, 3), 23: (11, 0),
                   24: (11, 0), 20: (7, 0), 21: (7, 0)}
#update


def get_pos(n):
    if n in sprite_pos_dict:
        return sprite_pos_dict[n], False
    else:
        return tile_pos_dict[n], True


# update
def read_tileset():
    path_map = r"data/img/mapsheet.png"
    path_sprite = r"data/img/enemysheet.png"
    path_map_img = np.asarray(cv2.cvtColor(cv2.imread(path_map), cv2.COLOR_BGR2RGB))  # HWC
    path_sprite_img = np.asarray(cv2.cvtColor(cv2.imread(path_sprite), cv2.COLOR_BGR2RGB))
    path_map_img = np.transpose(path_map_img, (2, 0, 1))  # CHW
    path_sprite_img = np.transpose(path_sprite_img, (2, 0, 1))  # CHW

    return torch.from_numpy(path_map_img), torch.from_numpy(path_sprite_img)
#update

# return a matching number ranging from 0 to 27
def lookup(x):
    return LOOKUP_TABLE[x]


def gen_rand_input():
    path = "datasets/scenes/train"
    files = os.listdir()
    idx = random.randint(0, len(files))
    filename = files[idx]
    path = os.path.join(path, filename)
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    ndarray = np.zeros((16, 16), dtype=np.uint8)
    lis = []
    for st in lines:
        # lookup: return a matching number ranging from 0 to 27
        lis.append(list(map(lookup, st)))
    for i, val in enumerate(lis):
        ndarray[i, :] = val

    return ndarray


# platform postprocess
def pf_postprocess(tensor):  # N C H W
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
# map [-1,1] to [0,27]
def tensor2txt(tensor):
    #  NCHW
    t = torch.squeeze(tensor, 1)  # n x h x w
    t = (t.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    t = (t * (len(REV_LOOKUP_TABLE) - 1)).type(torch.uint8)
    n, row, col = t.shape[0], t.shape[-2], t.shape[-1]
    for num in range(n):
        lines = []
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


'''
    tensor: (1,h,w) or (h,w)
    map: HWC
    sprite: HWC
    
    return: (h,w,3)
'''
def better_visualize(tensor, map, sprite):
    shape = tensor.shape
    h, w = shape[-2], shape[-1]
    tile_h, tile_w = 16, 16
    new_tensor = torch.round(tensor.mul(0.5).add(0.5).mul(len(REV_LOOKUP_TABLE)-1)).type(torch.uint8).squeeze()
    target = torch.zeros((3, h * tile_h, w * tile_w), dtype=torch.uint8)  # RGB CHW

    for row in range(h):
        for col in range(w):
            pos, is_map = get_pos(new_tensor[row][col].item())
            tstart_y = row * tile_h
            tend_y = tstart_y + tile_h
            tstart_x = col * tile_w
            tend_x = tstart_x + tile_w
            start_y = pos[0] * tile_h
            end_y = start_y + tile_h
            start_x = pos[1] * tile_w
            end_x = start_x + tile_w
            target[:, tstart_y:tend_y, tstart_x:tend_x] = map[:, start_y:end_y, start_x:end_x] if is_map else sprite[:, start_y:end_y, start_x:end_x]

    return target



def tensor2img(tensor, map, sprite, out_type=np.uint8, min_max=(-1, 1), nrow_=None):
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
            new_tensor[i] = better_visualize(tensor[i], map, sprite)
        img_np = make_grid(new_tensor, nrow=nrow_, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = better_visualize(tensor, map, sprite).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = better_visualize(tensor, map, sprite).numpy()
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
        n_img = len(tensor)  # N
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
        img_np = ((img_np + 1) * 127.5).round()  # (img_np + 1) / 2 * 255
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()


def postprocess(image, map, sprite):
    return [tensor2img(img, map, sprite) for img in image]


def set_seed(seed, gl_seed=0):
    """  set random seed, gl_seed used in worker_init_fn function """
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
    if seed >= 0 and gl_seed >= 0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_gpu(args, distributed=False, rank=0, has_mps=False):
    """ set parameter to gpu or ddp """
    if args is None:
        return None
    if distributed and isinstance(args, torch.nn.Module):
        return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True,
                   find_unused_parameters=True)
    # else:
    # 	if not has_mps:
    # 		return args.cuda()
    # 	else:
    # 		return args.to('mps')
    return args


def set_device(args, distributed=False, rank=0):
    """ set parameter to gpu or cpu """
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (set_gpu(item, distributed, rank) for item in args)
        elif isinstance(args, dict):
            return {key: set_gpu(args[key], distributed, rank) for key in args}
        else:
            args = set_gpu(args, distributed, rank)
    # elif torch.has_mps:
    # 	args = set_gpu(args, distributed, rank, True)
    return args

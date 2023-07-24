import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from core import util
from kaitaistruct import KaitaiStream
from io import BytesIO
from data.util.level import Level
import zlib
import pandas as pd
from tqdm import tqdm
import random

from .util.mask import (left_half_cropping_bbox, right_half_cropping_bbox, bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_txt_file(filename):
    return filename.endswith('.txt')


def make_txt_dataset(dir):
    if os.path.isfile(dir):
        txts = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        txts = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_txt_file(fname):
                    path = os.path.join(root, fname)
                    txts.append(path)
    return txts


# drop levels whose h<16 or w<16 out
# return levels
# level:
# data: ([obj])level_data  conditions: (int)theme (int)difficulty (int)gamestyle
def make_level_dataset(dir):
    levels = []
    if os.path.isfile(dir):
        assert dir.endswith('.parquet'), '%s is not a parquet file' % dir
        df = pd.read_parquet(dir)
        cnt = 0
        for idx, item in tqdm(df.iterrows(), desc='preprocessing %d samples' % df.shape[0], total=df.shape[0], position=0, leave=True):
            level_data = Level(KaitaiStream(BytesIO(zlib.decompress(item["level_data"]))))
            # filter out vertical levels and levels with subworld
            if level_data.subworld.object_count > 0 or level_data.overworld.orientation.value == 1:
                continue
            max_x = -1
            max_y = -1
            for obj in level_data.overworld.objects:
                x = obj.x // 160
                if x > max_x:
                    max_x = x
                y = obj.y // 160
                if y > max_y:
                    max_y = y
            level = {"data_id": item["data_id"], "level_data": level_data, "gamestyle": item["gamestyle"], "theme": item["theme"], "difficulty": item["difficulty"], "max_y": max_y, "max_x": max_x}
            levels.append(level)
            cnt += 1
            if cnt >= 100:
                print(len(levels))
                break

    else:
        # TODO: multiple parquet files
        pass

    return levels


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingTextDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[16, 16], loader=pil_loader):
        txts = make_txt_dataset(data_root)     # path list
        if data_len > 0:
            self.txts = txts[:int(data_len)]
        else:
            self.txts = txts
        self.tfs = transforms.Compose([
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
        ])
        self.root_dir = data_root
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        path = self.txts[idx]
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        ndarray = np.zeros((16, 16), dtype=np.uint8)
        lis = []
        for st in lines:
            # return a matching number ranging from 0 to 24
            lis.append(list(map(util.lookup, st)))
        for i, val in enumerate(lis):
            ndarray[i, :] = val
        img = ndarray.reshape(16, 16, 1)              # H W C
        # sample = np.repeat(sample, 3, axis=2)       # to 3 channel
        if self.tfs:
            img = self.tfs(img)

        ret = {}
        # path = self.txts[idx]
        # img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)  # img with noised part
        mask_img = img*(1. - mask) + mask                          # masked part set to 1

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        # ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['path'] = path.rsplit("/")[-1].replace(".txt", "")
        return ret

    def __len__(self):
        return len(self.txts)

    def get_mask(self):
        # return an (1,h,w) tensor with masked region set to 1, otherwise 0.
        if self.mask_mode == 'vertical':
            _type = np.random.randint(0, 2)
            if _type == 0:
                mask = bbox2mask(self.image_size, left_half_cropping_bbox())
            else:
                mask = bbox2mask(self.image_size, right_half_cropping_bbox())
        elif self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')

        return torch.from_numpy(mask).permute(2,0,1)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # numpy: H x W x C
        # torch: C x H x W
        # map to [0, 1]
        ret = (sample.astype(float) / 133).transpose((2, 0, 1))
        # ret = (sample.astype(float) / (len(util.REV_LOOKUP_TABLE) - 1)).transpose((2, 0, 1))   #HWC->CHW
        return torch.from_numpy(ret).float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # map from [0, 1] to [-1, 1]
        ret = (sample.add(-self.mean[0])).mul(1/self.std[0])
        return ret


class UncroppingLevelDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[16, 16], loader=pil_loader):
        levels = make_level_dataset(data_root)
        if data_len > 0:
            self.levels = levels[:int(data_len)]
        else:
            self.levels = levels
        self.tfs = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        # randomly crop 16x16 from the bottom row as dataset
        level = self.levels[index]
        # 1. generate a random x coordinate to start cropping
        if level['max_x'] <= self.image_size[0]:
            start_x = 0
        else:
            start_x = np.random.randint(level['max_x']-self.image_size[0])
        start_y = 0
        # 2. crop
        level_data = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        # level_data.fill(0)
        for obj in level['level_data'].overworld.objects:
            if start_x <= obj.x < start_x + self.image_size[0] and start_y <= obj.y < self.image_size[1]:
                # PROBLEM: no empty element
                # SOLUTION: add 1 to each element and keep empty element 0
                level_data[obj.x:obj.x+obj.width, obj.y:obj.y+obj.height] = obj.id.value + 1

        # 10 themes, 5 gamestyles, 4 difficulties
        # cond_theme = torch.zeros((10,), dtype=torch.long)
        # cond_gs = torch.zeros((5,), dtype=torch.long)
        # cond_df = torch.zeros((4,), dtype=torch.long)
        theme, gamestyle, difficulty = level['theme'], level['gamestyle'], level['difficulty']
        # cond_theme[theme] = 1
        # cond_gs[gamestyle] = 1
        # cond_df[difficulty] = 1
        cond = torch.tensor([theme, difficulty, gamestyle]).long()
        # cond = (torch.ones(1)*theme).long()

        level_data = level_data.reshape(16, 16, 1)
        if self.tfs:
            level_data = self.tfs(level_data)

        ret = {}

        img = level_data
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = level_data
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = level['data_id']

        return ret, cond

    def __len__(self):
        return len(self.levels)

    def get_mask(self):
        # return an (1,h,w) tensor with masked region set to 1, otherwise 0.
        if self.mask_mode == 'vertical':
            _type = np.random.randint(0, 2)
            if _type == 0:
                mask = bbox2mask(self.image_size, left_half_cropping_bbox())
            else:
                mask = bbox2mask(self.image_size, right_half_cropping_bbox())
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')

        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[16, 16], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)



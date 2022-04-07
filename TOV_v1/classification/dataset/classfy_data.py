# -*- coding: utf-8 -*-
'''
Dataset for Scene classification.
'''
import time
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
# from .transform import get_cls_transform as get_transform


def imread_with_cv(pth):
    ''' Only load RGB images data. '''
    img = cv2.imread(pth, 1)
    img = img[:, :, ::-1]  # BGR→RGB
    return img.copy()


def imread_with_pil(self, pth):
    ''' Only load RGB images data. '''
    with open(pth, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def shuffle_samples(samples, seed=None):
    if seed is None:
        seed = np.random.randint(100000)
    np.random.seed(seed)
    np.random.shuffle(samples)
    return samples


class ClassfyData(data.Dataset):
    '''
    Dataset loader for scene classification.
    Args:
        init_data - Dict of init data = {'train': [(img_path, lbl), ..], 'val': [...]}
        split - One of ['train', 'val', 'test']
        ratio - (float) extra parameter to conctrl the num of dataset
            if ratio < 0.01, then stand for nshort mode:
                ratio = 0.00n, where n is the num of sample per category
        transform - 可以传入自定义的transform对象
    '''
    def __init__(self,
                 init_data,
                 split='train',
                 dtype=['RGB'],
                 ratio=1,
                 transforms=None,
                 class_wise_sampling=True,
                 data_seed=0,
                 debug=False):
        # Set all input args as attributes
        self.__dict__.update(locals())

        if len(dtype) == 1 and dtype[0] == 'RGB':
            self.dtype = 'RGB'
            self.imread_func = imread_with_cv
        else:
            self.dtype = 'other'
            # self.imread_func = imread_with_gdal

        # Collect all dataset files, divide into dict.
        tic = time.time()
        self.imgs, self.lbls = [], []
        total_num = 0
        self.sample_statis = ''

        self.num_classes = len(init_data[split].keys())
        if type(init_data) is dict:
            if class_wise_sampling:
                for (cls, samples) in init_data[split].items():
                    total_num += len(samples)
                    if ratio <= 1:
                        use_num = max(round(len(samples)*ratio), 1)
                    else:
                        use_num = min(int(ratio), len(samples))
                    self.sample_statis += '| %s | %d |_' % (cls, use_num)
                    if use_num == 0:
                        continue
                    if data_seed:
                        samples = shuffle_samples(samples, data_seed)
                    for (pth, lbl) in samples[:use_num]:
                        self.imgs.append(pth)
                        self.lbls.append(lbl)
            else:
                categories = {}
                for (cls, samples) in init_data[split].items():
                    for (pth, lbl) in samples:
                        self.imgs.append(pth)
                        self.lbls.append(lbl)
                    total_num += len(samples)
                    categories[cls] = lbl
                if ratio <= 1:
                    use_num = max(round(total_num*ratio), 1)
                else:
                    use_num = min(int(ratio), total_num)
                self.shuffle_data(data_seed)  # shuffle data with seed
                self.imgs = self.imgs[:use_num]
                # print(self.imgs)
                self.lbls = self.lbls[:use_num]
                for cls, lbl in categories.items():
                    cls_num = (np.array(self.lbls) == lbl).sum()
                    self.sample_statis += '| %s | %d |_' % (cls, cls_num)
        self.use_num = len(self.imgs)

        print('{} set contains {} images, in {} categories.'.format(
              split, total_num, self.num_classes))
        if split == 'train':
            print('*'*6, f'data seed = {data_seed}', '*'*6)
            print(self.sample_statis)
        print('Actual number of samples used = {}'.format(self.use_num))

    def load_data_from_disk(self, index):
        img = self.imread_func(self.imgs[index])
        lbl = np.array(self.lbls[index], dtype=np.int64)
        return img, lbl

    def shuffle_data(self, seed=None):
        if seed is None:
            seed = np.random.randint(100000)

        np.random.seed(seed)
        np.random.shuffle(self.imgs)
        np.random.seed(seed)
        np.random.shuffle(self.lbls)

    def __len__(self):
        return self.use_num

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data_from_disk(index)

        img = Image.fromarray(img)
        # sample = {'image': img, 'label': lbl, 'name': self.imgs[index]}
        if self.transforms is not None:
            img = self.transforms(img)

        return img, lbl  # , self.imgs[index]

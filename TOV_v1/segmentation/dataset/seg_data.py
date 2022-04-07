# -*- coding: utf-8 -*-
'''
Dataset for segmentation.
'''
import time
import numpy as np
import cv2
from torch.utils import data

TYPE2BAND = {'RGB': 3, 'NIR': 1, 'SAR': 1}


def imread_with_cv(pth):
    ''' Only load RGB images data. '''
    img = cv2.imread(pth, 1)
    img = img[:, :, ::-1]  # BGR→RGB
    return img.copy()


class SegData(data.Dataset):
    '''
    Dataset loader for Segmentation.
    Args:
        init_data - Dict of init data = {'train': {'label': [], 'RGB': []}, 'val': [...]}
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
                 data_seed=0,
                 debug=False,
                 **kwargs):
        # Set all input args as attributes
        self.__dict__.update(locals())

        # Collect all dataset files, divide into dict.
        tic = time.time()
        self.imgs, self.lbls = {}, []

        total_num = len(init_data[split]['image'][dtype[0]])
        if ratio <= 1:
            use_num = max(round(total_num*ratio), 1)
        else:
            use_num = min(int(ratio), total_num)

        if type(init_data) is dict:
            for dt in dtype:
                self.imgs[dt] = init_data[split]['image'][dt]
            self.lbls = init_data[split]['label']

            self.shuffle_data(data_seed)  # shuffle data with seed

            for dt in self.imgs.keys():
                self.imgs[dt] = self.imgs[dt][:use_num]
            self.lbls = self.lbls[:use_num]
            # print(self.imgs)
        elif type(init_data) is str:
            raise NotImplementedError()
        self.use_num = len(self.imgs[dtype[0]])

        print('{} set contains {} images.'.format(split, total_num))
        if split == 'train':
            print('*'*6, f'data seed = {data_seed}', '*'*6)
        print('Actual number of samples used = {}'.format(self.use_num))
        print(self.imgs[dtype[0]][:5])

        print('Time to collect data = {:.2f}s.'.format(time.time()-tic))

    def imread(self, index):
        images = []
        for dt in self.dtype:
            img = cv2.imread(self.imgs[dt][index], cv2.IMREAD_LOAD_GDAL)
            if dt == 'RGB':
                img = img[:, :, ::-1]
            elif TYPE2BAND[dt] == 1:
                img = np.expand_dims(img, axis=2)
            images.append(img)
        image = np.concatenate(images, axis=2)
        return image

    def load_data_from_disk(self, index):
        img = self.imread(index)
        lbl = None
        if self.lbls is not None:
            lbl = cv2.imread(self.lbls[index], cv2.IMREAD_LOAD_GDAL)
        return img, lbl

    def shuffle_data(self, seed=None):
        if seed is None:
            seed = np.random.randint(100000)

        for dt in self.imgs.keys():
            np.random.seed(seed)
            np.random.shuffle(self.imgs[dt])
        np.random.seed(seed)
        np.random.shuffle(self.lbls)

    def __len__(self):
        return self.use_num

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data_from_disk(index)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']

        sample = {'image': img, 'label': lbl, 'name': self.imgs[self.dtype[0]]}
        return sample

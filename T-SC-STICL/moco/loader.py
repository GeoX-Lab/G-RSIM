# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import ot
from torchvision import transforms
from torchvision.transforms.transforms import Resize

IMG_EXTENSIONS = [
   '.jpg', '.JPG', '.jpeg', '.JPEG',
   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class TwoCropsTransform_sti:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, prob):
        self.base_transform = base_transform
        self.prob = prob

    def __call__(self, x1, x2):

        q = self.base_transform(x1)

        is_sttrans = np.random.rand()
        if is_sttrans < self.prob:
            xt = ST_transf(x1, x2)
            k = self.base_transform(xt)
        else:
            k = self.base_transform(x1)

        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def ST_transf(content_img, style_img):

    adjust_image = transforms.Compose([
        transforms.Resize([256, 256]),
        ])
    rand_seed = np.random.RandomState(42)

    content_img = adjust_image(content_img)
    style_img = adjust_image(style_img)
    
    I1 = np.array(content_img).astype(np.float64) / 256
    I2 = np.array(style_img).astype(np.float64) / 256
    X1 = I1.reshape((I1.shape[0] * I1.shape[1], I1.shape[2]))
    X2 = I2.reshape((I2.shape[0] * I2.shape[1], I2.shape[2]))
    idx1 = rand_seed.randint(X1.shape[0], size=(1000,))
    idx2 = rand_seed.randint(X2.shape[0], size=(1000,))
    Xs = X1[idx1, :]
    Xt = X2[idx2, :]

    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    trans_Xs_emd = ot_emd.transform(Xs=X1, batch_size=1024)
    Image_emd = np.clip(trans_Xs_emd.reshape(I1.shape), 0, 1)
    Image_emd = Image_emd * 255
    Image_aug = Image.fromarray(Image_emd.astype('uint8'))

    return Image_aug

def is_image_file(filename):
   return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
   classes = os.listdir(dir)
   classes.sort()
   class_to_idx = {classes[i]: i for i in range(len(classes))}
   return classes, class_to_idx


def make_dataset(dir, class_to_idx, num_sample = 0):
    images = []
    if num_sample > 0:
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            count = 0
            for filename in os.listdir(d):
                if is_image_file(filename) and count < num_sample:
                    path = '{0}/{1}'.format(target, filename)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    count = count + 1
    else:
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for filename in os.listdir(d):
                if is_image_file(filename):
                    path = '{0}/{1}'.format(target, filename)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def default_loader(path):
   return Image.open(path).convert('RGB')

class ImageFolderLoader(data.Dataset):
    def __init__(self, root, transform=None,
                target_transform=None,
                loader=default_loader, num_sample = 0, is_pretrain = False):

        if num_sample < 0:
            raise Exception('Error: Number of sample should not less than 0')

        if num_sample > 0:
            if is_pretrain:
                raise Exception('Error: Pretrain mode can not choose number of sample')
            else:
                classes, class_to_idx = find_classes(root)
                imgs = make_dataset(root, class_to_idx, num_sample)
        else: #num_sample is 0
            classes, class_to_idx = find_classes(root)
            imgs = make_dataset(root, class_to_idx)
        

        self.root = root
        self.is_pretrain = is_pretrain
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        if self.is_pretrain:

            path, target = self.imgs[index]
            max_index = len(self.imgs)

            #Random pick an image as scene sample
            index_aug = np.random.randint(0, max_index)
            path_aug, target_aug = self.imgs[index_aug]

            img = self.loader(os.path.join(self.root, path))
            img_aug = self.loader(os.path.join(self.root, path_aug))

            if self.transform is not None:
                img = self.transform(img, img_aug)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:

            path, target = self.imgs[index]
            img = self.loader(os.path.join(self.root, path))

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x1, x2):
        
        q = self.base_transform(x1)
        k = self.base_transform(x1)
        return [q, k]
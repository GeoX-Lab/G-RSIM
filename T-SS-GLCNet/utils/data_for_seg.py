# -*- coding:utf-8 -*-
'''
Dataset for GLCC.
    by QiJi

SegData - 主类（仅用于训练及训练期间的快速验证）
SegData_pre - 副类（仅用于正式验证和测试）
SegData_h5 - 读取hdf5数据，不建议使用

备注：
为了与之前版本dataset对应，增加了Train_Dataset用于训练（相当于主类SegData）
MyDataset_1用于quick val 和val

'''
import os

import cv2
# import h5py
import numpy as np
# import time
import torch
import tifffile
# import matplotlib.pyplot as plt
# from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from skimage import io
Type2Band = {'RGB': 3, 'NIR': 1, 'SAR': 1,'RGBIR':4}
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def get_label_info(file_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV or Txt format!

    Args:
        file_path: The file path of the class dictionairy

    Returns:
        Two lists: one for the class names and the other for the label values
    """
    import csv
    filename, exten = os.path.splitext(file_path)
    if not (exten == ".csv" or exten == ".txt"):
        return ValueError("File is not a CSV or TxT!")

    class_names, label_values = [], []
    with open(file_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)  # skip one line
        print(header)
        for row in file_reader:
            if row != []:
                class_names.append(row[0])
                label_values.append([int(row[1]),int(row[2]),int(row[3])])
    return class_names, label_values
def filelist(floder_dir, ifPath=False, extension=None,quick_train=True):
    '''
    Get names(or whole path) of all files(with specify extension)
    in the floder_dir and return as a list.

    Args:
        floder_dir: The dir of the floder_dir.
        ifPath:
            True - Return whole path of files.
            False - Only return name of files.(Defualt)
        extension: Specify extension to only get that kind of file names.

    Returns:
        namelist: Name(or path) list of all files(with specify extension)
    '''
    if extension is not None:
        if type(extension) != list:
            extension = [extension]

    namelist = sorted(os.listdir(floder_dir))

    if ifPath:
        for i in range(len(namelist)):
            namelist[i] = os.path.join(floder_dir, namelist[i])

    if extension is not None:
        n = len(namelist)-1  # orignal len of namelist
        for i in range(len(namelist)):
            if namelist[n-i].split('.')[-1] not in extension:
                namelist.remove(namelist[n-i])  # discard the files with other extension
    if quick_train:
        imglist=[]
        for i in range(len(namelist)):
            img=io.imread(namelist[i])
            imglist.append(img)

        return imglist
    else:
        return namelist

def filelist_fromtxt(floder_dir,txt_path, ifPath=True,ifarray=True):
    f=open(txt_path,'r')
    sourceInLines=f.readlines()
    f.close()
    namelist=[]
    # 定义一个空列表，用来存储结果
    for line in sourceInLines:
        img_name = line.strip('\n') # 去掉每行最后的换行符'\n'
        if ifPath:       
            img_name = os.path.join(floder_dir, img_name)
        namelist.append(img_name)
    namelist=sorted(namelist)
    if ifarray:
        imglist=[]
        for i in range(len(namelist)):
            img=io.imread(namelist[i])
            imglist.append(img)
        return imglist
    else:
        return namelist
    
   
def class_label(label, label_values):
    '''
    Convert RGB label to 2D [HW] array, each pixel value is the classified class key.
    '''
    semantic_map = np.zeros(label.shape[:2], label.dtype)
    for i in range(len(label_values)):
        equality = np.equal(label, label_values[i])
        class_map = np.all(equality, axis=-1)
        semantic_map[class_map] = i
    return semantic_map




def filp_array(array, flipCode):
    '''Filp an [HW] or [HWC] array vertically or horizontal according to flipCode.'''
    if flipCode != -1:
        array = np.flip(array, flipCode)
    elif flipCode == -1:
        array = np.flipud(array)
        array = np.fliplr(array)
    return array


def get_transform(opt, split):
    """ Get train/val transform for image. """
    if split == 'train':
        transforms = T.Compose([
            T.ColorJitter(brightness=0.2,
                          contrast=0.2,
                          saturation=0.2,
                          hue=0.05),
            T.ToTensor(),
            T.Normalize(opt.mean, opt.std),
        ])
    else:
        transforms = T.Compose([
            T.Resize(tuple(opt.input_size)),
            T.ToTensor(),
            T.Normalize(opt.mean, opt.std),
        ])

    return transforms


def random_crop_pair(image, label, crop_hw=(256, 256)):
    '''
    Crop image and label randomly

    '''
    crop_h, crop_w = crop_hw
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise Exception('Image and label must have the same shape')
    if (crop_h < image.shape[0] and crop_w < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_h)  # row
        y = np.random.randint(0, image.shape[1] - crop_w)  # column
        # label maybe multi-channel[H,W,C] or one-channel [H,W]
        return image[x:x + crop_h, y:y + crop_w], label[
            x:x + crop_h, y:y + crop_w]
    elif (crop_h == image.shape[0] and crop_w == image.shape[1]):
        return image, label
    else:
        raise Exception('Crop size > image.shape')


def randomShiftScaleRotate(image,
                           mask=None,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT,
                           p=0.5):
    """
    Random shift scale rotate image (support multi-band) may be with mask.

    Args:
        p (float): Probability of rotation.
    """
    if np.random.random() < p:
        if len(image.shape) > 2:
            height, width, channel = image.shape
        else:  # TODO: test
            (height, width), channel = image.shape, 1

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        if channel > 3:
            for c in range(channel):
                band = image[:, :, c]
                image[:, :, c] = cv2.warpPerspective(
                    band, mat, (width, height),
                    flags=cv2.INTER_LINEAR, borderMode=borderMode)
        else:
            image = cv2.warpPerspective(
                image, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
        if mask is not None:
            mask = cv2.warpPerspective(
                mask, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
    if mask is not None:
        return image, mask
    else:
        return image


def randomHueSaturationValue(image,
                             hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255),
                             u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0],
                                      hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image


def randpixel_noise(image, p=0.01):
    """Randomly perturbed the values of some pixels in the image [0,255].

    Args:
        p: The probability that each pixel is perturbed.
    """
    prob_mat = np.random.random(size=image.shape[:2])
    noise = np.random.randint(0, 256, size=image.shape, dtype=image.dtype)
    noise[prob_mat >= p] = 0

    noise_image = image.copy()
    noise_image[prob_mat < p, None] = 0
    noise_image += noise

    return noise_image


def join_transform(image, label, input_size):
    """
    Data augment function for training RS imagery with mutiple bands.
    Args:
        image: [HWC] ndarray of RS image with mutiple bands
        label: [HW1] ndarray of label
    """
    band_num = image.shape[2]
    # Random Flip
    f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
    if f != 2:
        image, label = filp_array(image, f), filp_array(label, f)
    # Random Roate (Only 0, 90, 180, 270)
    k = np.random.randint(0, 4)  # [0, 1, 2, 3]
    image = np.rot90(image, k, (1, 0))  # clockwise
    label = np.rot90(label, k, (1, 0))

    # Random Crop
    # image, label = join_randomScaleAspctCrop(image, label)
    image, label = random_crop_pair(image, label, input_size)
    image, label = randomShiftScaleRotate(
        image, label, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
        aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
    if band_num >= 3:
        image[:, :, :3] = randomHueSaturationValue(
            image[:, :, :3], hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
            val_shift_limit=(-15, 15))
        if band_num > 3:
            for c in range(3, band_num):
                image[:, :, c] = randomColorAugment(image[:, :, c], 0.01, 0.01)
                image[:, :, c] = randpixel_noise(image[:, :, c], p=0.001)
    elif band_num < 3:
        for c in range(0, band_num):
            image[:, :, c] = randomColorAugment(image[:, :, c], 0.01, 0.01)
            image[:, :, c] = randpixel_noise(image[:, :, c], p=0.001)

    return image, label


def slide_crop(image, crop_params=[256, 256, 128], pad_mode=0, ifbatch=False):
    '''
    Slide crop image(ndarray) into small piece, return list of sub_image or a [NHWC] array.

    Args:
        crop_params: [H, W, stride] (default=[256, 256, 128])

        pad_mode: One of the [-1, 0, 1] int values,
            -1 - Drop out the rest part;
            0 - Pad image at the end of W & H for fully cropping;
            1 - Pad image before and after W & H for fully cropping.
        ifbatch: Ture-return [NHWC] array; False-list of image.
    Return:
        size: [y, x]
            y - Num of images in the row direction.
            x - Num of images in the col direction.

    Carefully: If procided clipping parameters cannot completely crop the entire image,
        then it cannot be restored to original size during the recovery process.
    '''
    crop_h, crop_w, stride = crop_params
    dim = len(image.shape)
    h, w = image.shape[:2]

    h_drop, w_drop = (h-crop_h) % stride, (w-crop_w) % stride
    if h_drop or w_drop:
        # Cannot completely slide-crop the target image
        if pad_mode == 1:
            image, _ = pad_array(image, crop_h, crop_w, stride, stride)
        elif pad_mode == 0:
            pad_params = [(0, stride-h_drop), (0, stride-w_drop)]
            if dim == 3:
                pad_params.append((0, 0))
            image = np.pad(image, tuple(pad_params), 'constant')
        elif pad_mode == -1:
            image = image[:-h_drop, :-w_drop]
        h, w = image.shape[:2]  # Update image size

    image_list = []
    y = 0  # y:h(row)
    for i in range((h-crop_h)//stride + 1):
        x = 0  # x:w(col)
        for j in range((w-crop_w)//stride + 1):
            tmp_img = image[y:y+crop_h, x:x+crop_w].copy()
            if ifbatch:
                tmp_img = np.expand_dims(tmp_img, axis=0)
            image_list.append(tmp_img)
            x += stride
        y += stride
    size = [i+1, j+1]

    if ifbatch:
        batch_size = len(image_list)
        if batch_size == 1:
            return image_list[0], size
        else:
            image_bantch = np.squeeze(np.stack(image_list, axis=1))
            return image_bantch, size

    return image_list, size


def randomColorAugment(image, brightness=0.1, contrast=0.1):
    if brightness > 0:
        brightness_factor = np.random.uniform(max(0, 1-brightness), 1+brightness)
        if brightness_factor > 1:
            alpha = brightness_factor - 1
            degenerate = np.ones(image.shape, dtype=np.uint8) * 255
        elif brightness_factor <= 1:
            alpha = 1 - brightness_factor
            degenerate = np.zeros(image.shape, dtype=np.uint8)
        image = cv2.addWeighted(degenerate, alpha, image, (1-alpha), 0)

    # Adjust contrast, saturation and hue reference: https://zhuanlan.zhihu.com/p/24425116
    if contrast > 0:
        contrast_factor = np.random.uniform(max(0, 1-contrast), 1+contrast)
        image = np.clip(image * contrast_factor, 0, 255)
    return image


def join_randomScaleAspctCrop(image, label, scale=(0.25, 0.75), ratio=(3./4., 4./3.)):
    '''Crop the given Image(np.ndarray) to random size and aspect ratio,
    and finally resized to given size.
    Args:
        size: expected output size of each edge(default: 0.6 to 1.4 of original size)
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped(default: 3/4 to 4/3)
        interpolation: Default=PIL.Image.BILINEAR
    '''
    H, W = image.shape[:2]  # ori_height, ori_width
    area = H*W

    for attempt in range(10):
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.random.uniform(*ratio)

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))

        if np.random.random() < 0.5:
            w, h = h, w

        if w < W and h < H:
            i = np.random.randint(0, H - h)  # crop start point(row/y)
            j = np.random.randint(0, W - w)  # crop start point(col/x)
            image = image[i:i+h, j:j+w]
            label = label[i:i+h, j:j+w]
            return image, label
    # Fallback
    w = min(w, H)
    i, j = (H - w) // 2, (W - w) // 2
    image = image[i:i+h, j:j+w]
    label = label[i:i+h, j:j+w]
    return image, label


class SegData(data.Dataset):
    '''
    Pixel-level anntation Dataset for scene segmentation,
    load all normal image and label.
    Args:
        opt - configs about dataset.
            (opt.dataset_dir) - The dir of the train(test) dataset folder:
            ├── root
            |   ├── train
            |   ├── train_labels
            |   ├── val
            |   ├── val_labels
        split: 'train' or 'val
        transform: transform object or None(default) or
            'auto'(will make normal image transform according to train/eval)
    '''

    def __init__(
            self,
            root,
            split,
            opt,  # 包含了数据集路径
            ratio=1,
            transform=None):
        self.opt = opt
        self.root = opt.dataset_dir
        self.split = split
        self.ratio = ratio
        self.quick_train=opt.quick_train
        self.dtype = opt.dtype  # ['RGB', 'NIR', 'SAR']
        self.crop_mode = opt.crop_mode  # 'slide' or 'random'
        #self.bl_dtype = opt.bl_dtype
        self.insize = opt.input_size
        #self.class_value_list = opt.class_value_list
       # self.class_num = opt.class_num_total
        class_names, label_values = get_label_info(opt.class_dict_road)
        self.label_values=label_values
        split1 = None
        if 'train' in split:
            split1 = 'train'
        elif 'val' in split:
            split1 = 'val'
        else:
            raise Exception('error')

        # Collect all dataset files, divide into dict.
        print('Collecting %s kind of image data:' % self.dtype)
        self.names=filelist_fromtxt(self.root + '/' + split1 + '_lbl', self.root + '/' + split + '_lbl.txt',ifPath=True,ifarray=False)
        self.lbls = filelist_fromtxt(self.root + '/' + split1 + '_lbl', self.root + '/' + split + '_lbl.txt',ifPath=True,ifarray=self.quick_train)
        self.imgs = {}
        self.band = Type2Band[self.dtype]
        self.imgs=filelist_fromtxt(self.root + '/%s_%s' % (split1, self.dtype),self.root + '/%s_%s.txt' % (split, self.dtype), ifPath=True,ifarray=self.quick_train)

        assert len(self.lbls) == len(self.imgs)
        print('%s set contains %d images, a total of  categories.' %
              (split, len(self.lbls)) )#, len(opt.category.table)))
        print('Actual number of samples used = %d' % int(len(self.lbls) * self.ratio))

    def __len__(self):
        return int(len(self.lbls) * self.ratio)

    def need_crop(self, input_shape):
        '''Judge if the input image needs to be cropped'''
        crop_H = self.insize[0]
        crop_W = self.insize[1]
        return input_shape[0] > crop_H and input_shape[1] > crop_W
    


class MyDataset_1(SegData):
    def __getitem__(self, index):
        data_dict = dict()

        if self.quick_train:
            lbl = self.lbls[index]
            image=self.imgs[index]
        else:
            lbl = io.imread(self.lbls[index])
            image = io.imread(self.imgs[index])
        if len(lbl.shape)==3:
            lbl=class_label(lbl, self.label_values)
        '''
        new_classvalue = 1
        for c in range(1, self.class_num):
            if c not in self.class_value_list:
                lbl[lbl == c] = 0
            else:
                lbl[lbl == c] = new_classvalue
                new_classvalue += 1
        
        '''
        lbl[lbl == 255] = 0 
        



        if SegData.need_crop(self, image.shape):
            if self.crop_mode == 'random':
                lbl = np.expand_dims(lbl, axis=2)
                image, lbl = random_crop_pair(image, lbl, self.insize)
                image = image.astype(np.float32).transpose(2, 0, 1).copy() / 255.0 * 3.2 - 1.6
                #lbl = torch.from_numpy(np.squeeze(lbl).copy()).long()
            elif self.crop_mode == 'slide':
                images, data_dict['crop_info'] = slide_crop(image, self.opt.crop_params, 0)
            # cv2.imshow('RGB', images[0][:, :, :3])
            # cv2.imshow('NIR', images[0][:, :, 3])
            # cv2.waitKey(0)
                images = [
                im.astype(np.float32).transpose(
                    2, 0, 1).copy() / 255.0 * 3.2 - 1.6 for im in images]
                image = np.stack(images)
        else:
            image = image.astype(np.float32).transpose(2, 0, 1).copy() / 255.0 * 3.2 - 1.6
        lbl = torch.from_numpy(np.squeeze(lbl).copy()).long()
        data_dict['name'] = os.path.basename(self.names[index])
        data_dict['image'], data_dict['label'] = image, lbl
        return data_dict


class Train_Dataset(SegData):
    '''
    wei

    '''
    def __getitem__(self, index):
        data_dict = dict()


        if self.quick_train:
            lbl = self.lbls[index]
            image = self.imgs[index]
        else:
            lbl = io.imread(self.lbls[index])
            image = io.imread(self.imgs[index])
        if len(lbl.shape)==3:
            lbl=class_label(lbl, self.label_values)
        '''
        # set new class value
        new_classvalue = 1
        for c in range(1, self.class_num):
            if c not in self.class_value_list:
                lbl[lbl == c] = 0
            else:
                lbl[lbl == c] = new_classvalue
                new_classvalue += 1
        '''
        lbl[lbl == 255] = 0 

        lbl = np.expand_dims(lbl, axis=2)
        image, lbl = join_transform(image, lbl, self.insize)
        
            
        
           
        image = image.astype(np.float32).transpose(2, 0, 1).copy() / 255.0 * 3.2 - 1.6
        lbl = torch.from_numpy(np.squeeze(lbl).copy()).long()

        data_dict['name'] = os.path.basename(self.names[index])
        data_dict['image'], data_dict['label'] = image, lbl
        return data_dict


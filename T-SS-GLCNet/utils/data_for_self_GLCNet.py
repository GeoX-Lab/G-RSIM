# -*- coding:utf-8 -*-
'''
data_seg_for_self_supervised_GLCNet
'''
import os
import cv2
import numpy as np
import torch
import tifffile
from torch.utils import data
from torchvision import transforms
from skimage import io
Type2Band = {'RGB': 3, 'NIR': 1, 'SAR': 1,'RGBIR':4}
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from PIL import ImageFilter
import random
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

def filelist(floder_dir, ifPath=False, extension=None):
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
    imglist=[]
    for i in range(len(namelist)):
        img=io.imread(namelist[i])
        imglist.append(img)
    
    return imglist

def filelist_fromtxt(floder_dir,txt_path, ifPath=True,ifarray=True):
    f=open(txt_path,'r')
    sourceInLines=f.readlines()
    f.close()
    namelist=[]

    for line in sourceInLines:
        img_name = line.strip('\n')
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

def filp_array(array, flipCode):
    '''Filp an [HW] or [HWC] array vertically or horizontal according to flipCode.'''
    if flipCode != -1:
        array = np.flip(array, flipCode)
    elif flipCode == -1:
        array = np.flipud(array)
        array = np.fliplr(array)
    return array



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))#ImageFilter.GaussianBlur高斯模糊
        return x

def get_transform():
    """ Get train/val transform for image. """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),#p的概率进行某transform操作，
            transforms.RandomGrayscale(p=0.2),#依概率 p 将图片转换为灰度图？
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    TwoCropsTransform(transforms.Compose(augmentation))
    return TwoCropsTransform(transforms.Compose(augmentation))


def random_crop_pair(image,  crop_hw=(256, 256)):
    '''
    Crop image and label randomly

    '''
    crop_h, crop_w = crop_hw
    
    if (crop_h < image.shape[0] and crop_w < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_h)  # row
        y = np.random.randint(0, image.shape[1] - crop_w)  # column
        # label maybe multi-channel[H,W,C] or one-channel [H,W]
        return image[x:x + crop_h, y:y + crop_w]
    elif (crop_h == image.shape[0] and crop_w == image.shape[1]):
        return image
    else:
        raise Exception('Crop size > image.shape')
def random_crop(image, crop_hw=(256, 256)):
    '''
    Crop image and label randomly

    '''
    crop_h, crop_w = crop_hw

    if (crop_h < image.size[0] and crop_w < image.size[1]):
        x = np.random.randint(0, image.size[0] - crop_h)  # row
        y = np.random.randint(0, image.size[1] - crop_w)  # column
        box1 = (x, y,x + crop_h, y + crop_w) 
        # label maybe multi-channel[H,W,C] or one-channel [H,W]
        return image.crop(box1)
    elif (crop_h == image.size[0] and crop_w == image.size[1]):
        return image
    else:
        raise Exception('Crop size > image.shape')
def random_resized_crop0(image,label=None,label1=None,size=(256,256),scale_limit=(0.08,1.0)):
    
    if len(image.shape) > 2:
        height, width, channel = image.shape
    else:  # TODO: test
        (height, width), channel = image.shape, 1
    out_img=np.zeros((size[0], size[1], channel), dtype=np.uint8)
    scale = np.random.uniform(scale_limit[0], scale_limit[1])
    crop_h=int(scale*height)
    crop_w =int(scale*width)
    if (crop_h < image.shape[0] and crop_w < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_h)  # row
        y = np.random.randint(0, image.shape[1] - crop_w)  # column
        if label is not None:
            if channel>3:
                for i in range(channel):
                    out_img[:,:,i]=cv2.resize(image[x:x + crop_h, y:y + crop_w,i],size)
            else:
                out_img=cv2.resize(image[x:x + crop_h, y:y + crop_w],size)
            return out_img, cv2.resize(label[x:x + crop_h, y:y + crop_w],size), cv2.resize(label1[x:x + crop_h, y:y + crop_w],size)
        else:
            if channel>3:
                for i in range(channel):
                    out_img[:,:,i]=cv2.resize(image[x:x + crop_h, y:y + crop_w,i],size)
            else:
                out_img=cv2.resize(image[x:x + crop_h, y:y + crop_w],size)
            return out_img
    #cv2.resize(img[:,:,0], (256, 256))

def random_resized_crop(image,label,size=(224,224), scale=(0.25, 0.75), ratio=(3./4., 4./3.)):
    '''Crop the given Image(np.ndarray) to random size and aspect ratio,
    and finally resized to given size.
    Args:
        size: expected output size of each edge(default: 0.6 to 1.4 of original size)
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped(default: 3/4 to 4/3)
        interpolation: Default=PIL.Image.BILINEAR
    '''
    if len(image.shape) > 2:
        height, width, channel = image.shape
    else:  # TODO: test
        (height, width), channel = image.shape, 1
    out_img=np.zeros((size[0], size[1], channel), dtype=np.uint8)
   
    
           
    H, W = image.shape[:2]  # ori_height, ori_width
    area = H*W
    flag=True
    for attempt in range(5):
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
            label=label[i:i+h,j:j+w]
            flag=False
            break

           
    if flag:
        w = min(w, H)
        i, j = (H - w) // 2, (W - w) // 2
        image = image[i:i+h, j:j+w]
        label=label[i:i+h,j:j+w]
    
    if channel>3:
        label=cv2.resize(label,size, interpolation=cv2.INTER_NEAREST)
        for i in range(channel):
            out_img[:,:,i]=cv2.resize(image[:,:,i],size)
    else:
        out_img=cv2.resize(image,size)
        label=cv2.resize(label,size, interpolation=cv2.INTER_NEAREST)
    return out_img,label
  
def randomShiftScaleRotate(image,
                           mask=None,mask1=None,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT,
                           p=0.5):
    """
    随机平移缩放旋转
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
                flags=cv2.INTER_LINEAR, borderMode=borderMode)#cv2.warpPerspective()透视变换函数，mat变换矩阵，(width, height)输出大小
        if mask is not None:
            mask = cv2.warpPerspective(
                mask, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
            mask1 = cv2.warpPerspective(
                mask1, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
    if mask is not None:
        return image, mask,mask1
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


def join_transform(image,  label1,input_size=(224,224)):
    """
    Data augment function for training RS imagery with mutiple bands.
    Args:
        image: [HWC] ndarray of RS image with mutiple bands
        label: [HW1] ndarray of label
    """
    if image.shape[1]>256:
        image = random_crop_pair(image,  [256,256])
    image1=image.copy()
    #label1=np.array(range(256*256)).reshape(256,256)
    #label1 = np.expand_dims(label1, axis=2)
    label2 = label1.copy()
    image,label1 = random_resized_crop(image,label1,size=input_size,scale=(0.4,1.0))
    image1,label2 = random_resized_crop(image1,label2,size=input_size,scale=(0.4,1.0))
    band_num = image.shape[2]
    # Random Flip
    f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
    if f != 2:
        image1= filp_array(image1, f)
        label2=filp_array(label2,f)
    #f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
    #if f != 2:
        #image1= filp_array(image1, f)
    #transforms.RandomApply([
                #transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),#p的概率进行某transform操作，
           
            
    # Random Roate (Only 0, 90, 180, 270)
    if np.random.random() < 0.8:
        k = np.random.randint(0, 4)  # [0, 1, 2, 3]
        image1 = np.rot90(image1, k, (1, 0))  # clockwise
        label2 = np.rot90(label2, k, (1, 0))
    
    #k = np.random.randint(0, 4)  # [0, 1, 2, 3]
    #image1 = np.rot90(image1, k, (1, 0))  # clockwise
    if np.random.random() < 0.8:
        image1=color_aug(image1)
    #if np.random.random() < 0.8:
        #image=color_aug(image)
    #image=random_image_to_gray(image,p=0.1)
    image1=random_image_to_gray(image1,p=0.1)
    if np.random.random() < 0.5:
        image1=random_GaussianBlur(image1)
   
    return image,image1,label1,label2

def join_transform_v1(image,  input_size=(224,224)):
    """
    Data augment function for training RS imagery with mutiple bands.
    Args:
        image: [HWC] ndarray of RS image with mutiple bands
        label: [HW1] ndarray of label
    """
    image = random_crop_pair(image,  [256,256])
    image1=image.copy()
    label1=np.array(range(256*256)).reshape(256,256)
    label1 = np.expand_dims(label1, axis=2)
    label2=np.array(range(256*256)).reshape(256,256)
    label2 = np.expand_dims(label2, axis=2)
   # print(image.shape)
    image,label1 = random_resized_crop(image,label1,size=input_size,scale=(0.4,1.0))
    image1,label2 = random_resized_crop(image1,label2,size=input_size,scale=(0.4,1.0))
    band_num = image.shape[2]
    # Random Flip
    f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
    if f != 2:
        image1= filp_array(image1, f)
        label2=filp_array(label2,f)
    
           
            
    # Random Roate (Only 0, 90, 180, 270)
    if np.random.random() < 0.8:
        k = np.random.randint(0, 4)  # [0, 1, 2, 3]
        image1 = np.rot90(image1, k, (1, 0))  # clockwise
        label2 = np.rot90(label2, k, (1, 0))
    
    #k = np.random.randint(0, 4)  # [0, 1, 2, 3]
    #image1 = np.rot90(image1, k, (1, 0))  # clockwise
    if np.random.random() < 0.25:
        band_num=image1.shape[2]
        for c in range(0, band_num):
            image1[:, :, c] = randomColorAugment(image1[:, :, c], 0.2, 0.2)
    if np.random.random() < 0.25:
        band_num=image1.shape[2]
        for c in range(0, band_num):
            if band_num >= 3:
                image1[:, :, :3] = randomHueSaturationValue(image1[:, :, :3], hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
                                  val_shift_limit=(-15, 15))
    if np.random.random() < 0.25:
        band_num=image1.shape[2]
        for c in range(0, band_num):
            p=random.uniform(0,0.1)
            image1[:, :, c] = randpixel_noise(image1[:, :, c], p=p)
        
    
    image1=random_image_to_gray(image1,p=0.1)
    if np.random.random() < 0.25:
        image1=random_GaussianBlur(image1)
        
   
    return image,image1,label1,label2    
def color_aug(image):
    band_num=image.shape[2]
    
    if band_num >= 3:
        image[:, :, :3] = randomHueSaturationValue(
            image[:, :, :3], hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
            val_shift_limit=(-15, 15))
        if band_num > 3:
            for c in range(0, band_num):
                image[:, :, c] = randomColorAugment(image[:, :, c], 0.1, 0.1)
                image[:, :, c] = randpixel_noise(image[:, :, c], p=0.001)
    elif band_num < 3:
        for c in range(0, band_num):
            image[:, :, c] = randomColorAugment(image[:, :, c], 0.1, 0.1)
            image[:, :, c] = randpixel_noise(image[:, :, c], p=0.001)
    return image
def random_GaussianBlur(img,scale=(1,3)):
    k = np.random.randint(scale[0], scale[1]+1)
    k=k*2+1
    channel=img.shape[2]
    if channel>3:
        for i in range(channel):
            img[:,:,i]=cv2.GaussianBlur(img[:,:,i],(k,k),0)
    else:
        img=cv2.GaussianBlur(img,(k,k),0)
    return img
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

def random_image_to_gray(image, p=0.1):
    #随机灰度化（仅保留其中一个通道）
    
    if np.random.random() < p :
        n_channel=image.shape[2]
        t=np.random.randint(n_channel)
        for i in range(n_channel):
            image[:,:,i]=image[:,:,t]

    return image
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
        self.patch_size=opt.patch_size
        self.patch_num=opt.patch_num
        self.root = opt.dataset_dir
        self.split = split
        self.ratio = ratio
        self.dtype = opt.dtype
        self.quick_train=opt.quick_train
        self.crop_mode = opt.crop_mode  # 'slide' or 'random'
        self.bl_dtype = opt.bl_dtype
        self.insize = opt.input_size

        #self.transform=get_transform()
        self.lbl=np.expand_dims(np.array(range(256*256)).reshape(256,256), axis=2)
        # Collect all dataset files, divide into dict.
        print('Collecting %s kind of image data:' % self.dtype)
        #self.lbls = filelist(self.root + '/' + split + '_lbl', ifPath=True)
        #self.lbls1 = filelist(self.root + '/' + split + '_lbl1', ifPath=True)
        self.imgs = {}
        self.band = 0
        split1=None
        if 'train' in split:
            split1='train'
        elif 'val' in split:
            split1='val'
        else:
            raise Exception('error')
        #for dt in self.dtype:
            #print('\t', dt)
        self.band += Type2Band[self.dtype]
        self.imgs =filelist_fromtxt(self.root + '/%s_%s' % (split1, self.dtype),self.root + '/%s_%s.txt' % (split, self.dtype), ifPath=True,ifarray=self.quick_train)# filelist(self.root + '/%s_%s' % (split, dt), ifPath=True)
            #assert len(self.lbls) == len(self.imgs[dt])
        print('%s set contains %d images, a total of  categories.' %
              (split, len(self.imgs)) )#, len(opt.category.table)))
        #print('Actual number of samples used = %d' % int(len(self.lbls) * self.ratio))

    def __len__(self):
        return int(len(self.imgs) * self.ratio)

    def need_crop(self, input_shape):
        '''Judge if the input image needs to be cropped'''
        crop_H = self.insize[0]
        crop_W = self.insize[1]
        return input_shape[0] > crop_H and input_shape[1] > crop_W

class Train_Dataset(SegData):
    '''
    wei

    '''
    def __getitem__(self, index):
        data_dict = dict()

        #lbl = io.imread(self.lbls[index])
        #lbl1 = io.imread(self.lbls1[index])
        # set new class value
        if self.quick_train:
            image=self.imgs[index]
        else:
            image=io.imread(self.imgs[index])
        image,image1,lbl1,lbl2=join_transform(image,self.lbl.copy())
        image = image.astype(np.float32).transpose(2, 0, 1).copy() / 255.0 * 3.2 - 1.6
        image1 = image1.astype(np.float32).transpose(2, 0, 1).copy() / 255.0 * 3.2 - 1.6
        lbl1=np.squeeze(lbl1.astype(np.float32))
        lbl2 = np.squeeze(lbl2.astype(np.float32))
        index=get_index(lbl1,lbl2,(self.patch_size,self.patch_size),self.patch_num)
        index=torch.from_numpy(index).long()

        #lbl1 = torch.from_numpy(lbl1.copy()).long()
        #lbl2 = torch.from_numpy(lbl2.copy()).long()
        #for dt in self.dtype:
            #data_dict['name'] = os.path.basename(self.imgs[dt][index])
        data_dict['image'],data_dict['image1'] = image,image1
        #data_dict['label'], data_dict['label1'] = lbl1, lbl2
        data_dict['index'] = index
        return data_dict

def get_index(label1,label2,patch_size=(16,16),patch_num=4):
    index_result = np.zeros((patch_num, 4))
    index_i = 0
    range_x = patch_size[0] // 2
    range_x1 = label1.shape[0] - patch_size[0] // 2
    range_y = patch_size[1] // 2
    range_y1 = label1.shape[1] - patch_size[1] // 2
    # flag_early_stop = False

    list_for_select = label1[range_x:range_x1, range_y:range_y1].reshape(-1).tolist()

    list2 = label2[range_x:range_x1, range_y:range_y1].reshape(-1).tolist()
    # list2 = set(list2)  # this reduces the lookup time from O(n) to O(1)
    # list_for_select = [item for item in set(list_for_select) if item in set(list2)]
    # list2=set(list2)

    list_for_select = list(set(list_for_select).intersection(list2))
    # del list2

    # print(datetime.datetime.now())
    # one_segment1 = np.zeros((label1.shape[0], label1.shape[1]))
    # t = 1
    for i in range(patch_num):
        # flag = True
        # while flag:

       # if len(list_for_select) <= 1:
            # flag_early_stop = True
           # break
        a = random.sample(list_for_select, 1)
        target1_index = np.argwhere(label1 == a)
        if len(target1_index.shape) == 2:
            if target1_index[0][0] - patch_size[0] // 2 < 0 or target1_index[0][0] + patch_size[0] // 2 > \
                    label1.shape[0] or target1_index[0][1] - patch_size[1] // 2 < 0 or target1_index[0][1] + \
                    patch_size[1] // 2 > label1.shape[1]:
                for i1 in range(1, target1_index.shape[0]):
                    if target1_index[i1][0] - patch_size[0] // 2 < 0 or target1_index[i1][0] + patch_size[
                        0] // 2 > label1.shape[0] or target1_index[i1][1] - patch_size[1] // 2 < 0 or \
                            target1_index[i1][1] + patch_size[1] // 2 > label1.shape[1]:
                        continue
                    else:
                        target1_index = target1_index[i1, :]
                        break
            else:
                target1_index = target1_index[0, :]
        target2_index = np.argwhere(label2 == a)
        if len(target2_index.shape) == 2:
            if target2_index[0][0] - patch_size[0] // 2 < 0 or target2_index[0][0] + patch_size[0] // 2 > \
                    label2.shape[0] or target2_index[0][1] - patch_size[1] // 2 < 0 or target2_index[0][1] + \
                    patch_size[1] // 2 > label2.shape[1]:
                for i1 in range(1, target2_index.shape[0]):
                    if target2_index[i1, 0] - patch_size[0] // 2 < 0 or target2_index[i1][0] + patch_size[
                        0] // 2 > label2.shape[0] or target2_index[i1][1] - patch_size[1] // 2 < 0 or \
                            target2_index[i1][1] + patch_size[1] // 2 > label2.shape[1]:
                        continue
                    else:
                        target2_index = target2_index[i1, :]
                        break
            else:
                target2_index = target2_index[0, :]
        index_result[index_i, :] = [target1_index[0], target1_index[1], target2_index[0], target2_index[1]]
        index_i = index_i + 1
        t_list = label1[target1_index[0] - patch_size[0] // 2:target1_index[0] + patch_size[0] // 2,
                 target1_index[1] - patch_size[1] // 2:target1_index[1] + patch_size[1] // 2].reshape(-1).tolist()
        '''
        one_segment1[target1_index[0] - patch_size[0] // 2:target1_index[0] + patch_size[0] // 2,
        target1_index[1] - patch_size[1] // 2:target1_index[1] + patch_size[1] // 2] = t
        b1 = np.where(one_segment1 == t)
        t = t + 1
        b1 = b1[0] * label1.shape[0] + b1[1]
        t_list = label1.reshape(-1)[b1].tolist()
        '''

        # t_list = set(t_list)
        # list_for_select = set(list_for_select)
        # print('6')
        # print(datetime.datetime.now())

        list_for_select1 = set(list_for_select).difference(t_list)
        if len(list_for_select1)>1:
            list_for_select=list_for_select1
        # list_for_select=list(list_for_select - t_list)
        # list_for_select = [i for i in list_for_select if i not in t_list]
        # flag = False


    return index_result


'''
l=np.array(range(256*256)).reshape(256,256)
l= np.expand_dims(l, axis=2)
patch_size=(16,16)
print(datetime.datetime.now())
range_x=patch_size[0] // 2
range_x1=a.shape[0]-patch_size[0] // 2
range_y=patch_size[1] // 2
range_y1=a.shape[1]-patch_size[1] // 2
print(datetime.datetime.now())
#label1=l.copy()
'''
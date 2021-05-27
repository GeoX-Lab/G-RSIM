'''
Tools collection of Data(np.ndarray) processing.

Note:
    1. cv2.imwrite(image_name, image, [1, 100]),
        [1, 100] mean set [cv2.IMWRITE_JPEG_QUALITY, 100]

Version 1.0  2018-04-03 15:44:13 by QiJi
Version 1.5  2018-04-07 22:34:53 by QiJi
Version 2.0  2018-10-25 16:59:23 by QiJi
'''
import os

# import sys
import cv2
import numpy as np
from tqdm import tqdm

from dl_tools.basictools.dldata import get_label_info
from dl_tools.basictools.fileop import filelist, rename_file


# **********************************************
# ************ Image basic tools ***************
# **********************************************
def load_image(path, RGB=0):
    '''Unchange mode to load image(and may be convert it to RGB)
    Args:
        path: The whole path of image.
        RGB: 0-BGR(opencv deafault); 1-RGB(use cvtColor convert it to RGB)
    Returns:
        image: image
    '''
    image = cv2.imread(path, -1)  # BGR
    if RGB:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def convert_image_type(image_dir, new_type, old_type=None):
    '''Covert image's type(may be specify the old type).
    Args:
        new_type: The target type of image conversion(such as: 'png', 'jpg').
    '''
    image_names = filelist(image_dir, True, old_type)
    for name in tqdm(image_names):
        img = cv2.imread(name, 1)  # 默认BGR模式读，适应Tiff的标签图
        os.remove(name)
        name = os.path.splitext(name)[0]+'.'+new_type
        cv2.imwrite(name, img, [1, 100])


def one_hot_1(label, class_num):
    '''One hot code the label, not support RBG label.
    Args:
        label: a [HW] or [HW1] array
        class_num: num of classes
    Returns:
        one_hot: a 3D array of one_hot label (dtype=np.uint8), C=num of classes
    '''  
    one_hot = np.zeros([label.shape[0], label.shape[1], class_num], dtype=np.uint8)
    # one_hot = np.zeros([2616, 3860, 4], dtype=np.uint8)
    #  one_hot = np.zeros((a, b, c), dtype=np.uint8)
    for i in range(class_num):
        one_hot[:, :, i] = (label == i)  # TODO: need test [HW1],#此时为[HWC],C=6
    # img_path1 = 'D:/Data_Lib/Seg/MeiTB/one_hot'
    # if not os.path.exists(img_path1):
    #                 os.mkdir(img_path1)
    # cv2.imwrite(img_path1 + '/' + 'one' + '.tif', one_hot[:, :, 0:4])
    return one_hot



def one_hot_2(label, label_values):
    """
    One hot code the RBG label by replacing each pixel value with a vector of length num_classes
    Note: If class_dict is RGB, the label must be RGB(not BGR).
    Args:
        label: a [HWC] array, and C=3(be carefull about RGB or BGR)
        label_values: A list per class's color values
    Returns:
        one_hot: one_hot label, C=num of classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def one_hot_3(label, class_num):
    '''One hot code the label, classification result.
    Args:
        label: a [1] or [N1] array
        class_num: num of classes
    Returns:
        one_hot: a [NC] array of one_hot label, C=num of classes
    '''
    one_hot = np.zeros([label.shape[0], class_num], dtype=label.dtype)
    for i in range(class_num):
        one_hot[:, i] = (label == i)
    return one_hot


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


def reverse_one_hot(one_hot):
    '''Transform a 3D array in one-hot format (depth is num_classes),
    to a 2D array, each pixel value is the classified class key.
    '''
    return np.argmax(one_hot, axis=2)  # set axis=2 to limit the input is 3D


def colour_code_label(label, label_values, add_image=None, save_path=None):
    '''
    Given a [HW] array of class keys(or one hot[HWC]), colour code the label;
    also can weight the colour coded label and image, maybe save the final result.

    Args:
        label: single channel array where each value represents the class key.
        label_values
    Returns:
        Colour coded label or just save image return none.
    '''
    # TODO: 贼强，不过还没弄明白
    label, colour_codes = np.array(label), np.array(label_values)
    if len(label) == 3:
        label = np.argmax(label, axis=2)  # [HWC] -> [HW]
    color_label = colour_codes[label.astype(int)]  # TODO:此处直接uint8有错误
    color_label = color_label.astype(np.uint8)  # 把label中的每一个元素换成label_value（此时label_value可以是任意维数）

    if add_image is not None:
        if add_image.shape != color_label.shape:
            cv2.resize(color_label, (add_image.shape[1], add_image.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        add_image = cv2.addWeighted(add_image, 0.7, color_label, 0.3, 0)
        if save_path is None:
            return color_label, add_image

    if save_path is not None:
        cv2.imwrite(save_path, color_label, [1, 100])
        if add_image is not None:
            cv2.imwrite(rename_file(save_path, addstr='mask'), add_image, [5, 100])
        return  # no need to return label if saved

    return color_label


def mask_img(image, label, mask_value=[0, 0, 0]):
    '''Mask image with a specified value of label.
    Note: mask_value is BGR list
    '''
    equality = np.equal(label, mask_value)
    mask = np.all(equality, axis=-1)
    image[mask] = [0, 0, 0]
    return image


# **********************************************
# ******** Common data Pre-treatment ***********
# **********************************************
def slide_crop(image, crop_params=[256, 256, 128], ifPad=True, ifbatch=False):
    '''
    Slide crop image into small piece, return list of sub_image or a [NHWC] array.
    Args:
        crop_params: [H, W, stride] (default=[256, 256, 128])
        ifbatch: Ture-return [NHWC] array; False-list of image.
    Return:
        size: [y, x]
            y: Num of images in the row direction.
            x: Num of images in the col direction.
    Carefully: If procided clipping parameters cannot completely crop the entire image,
        then it cannot be restored to original size during the recovery process.
    '''
    crop_h, crop_w, stride = crop_params
    h, w = image.shape[:2]

    if (h-crop_h) % stride or (w-crop_w) % stride:
        # Cannot completely slide-crop the target image
        if ifPad:
            image, _ = pad_img(image, crop_h, crop_w, stride, stride)
            h, w = image.shape[:2]  # Update image size
        else:
            ValueError(
                'Provided crop-parameters [%d, %d, %d]' % (crop_h, crop_w, stride)
                + 'cannot completely slide-crop the target image')

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


def pad_img(image, kh, kw, sh, sw):
    '''Pad image according kernel size and stride.
    Args:
        image - array
        kh, kw - kernel height, kernel width
        sh, sw - height directional stride, width directional stride.
    '''
    h, w = image.shape[:2]
    d = len(image.shape)
    pad_h, pad_w = sh - (h-kh) % sh, sw - (w-kw) % sw
    pad_h = (pad_h//2, pad_h//2+1) if pad_h % 2 else (pad_h//2, pad_h//2)
    pad_w = (pad_w//2, pad_w//2+1) if pad_w % 2 else (pad_w//2, pad_w//2)
    pad_params = (pad_h, pad_w) if d == 2 else (pad_h, pad_w, (0, 0))
    return np.pad(image, pad_params, mode='constant'), pad_params


def resized_crop(image, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
    '''Crop the given PIL Image and resize it to desired size.
    Args:
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size: (Height, Width) must be tuple
    '''
    image = image[i:i+h, j:j+w]
    image = cv2.resize(image, size[::-1], interpolation)
    return image


def random_crop(image, crop_height=256, crop_width=256):
    '''
    Crop image(label) randomly
    '''
    if (crop_height < image.shape[0] and crop_width < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_height)  # row
        y = np.random.randint(0, image.shape[1] - crop_width)  # column
        return image[x:x + crop_height, y:y + crop_width]

    elif (crop_height == image.shape[0] and crop_width == image.shape[1]):
        return image

    else:
        raise Exception('Crop size > image.shape')


def random_crop_2(image, label, crop_height=256, crop_width=256):
    '''
    Crop image and label randomly

    '''
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise Exception('Image and label must have the same shape')
    if (crop_height < image.shape[0] and crop_width < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_height)  # row
        y = np.random.randint(0, image.shape[1] - crop_width)  # column
        # label maybe multi-channel[H,W,C] or one-channel [H,W]
        return image[x:x + crop_height, y:y + crop_width], label[
                x:x + crop_height, y:y + crop_width]
    elif (crop_height == image.shape[0] and crop_width == image.shape[1]):
        return image, label
    else:
        raise Exception('Crop size > image.shape')


def random_crop_3(image, label, crop_h=256, crop_w=256, num_classes=0):
    '''
    Crop image(label) randomly for serval times(now four times most),
    and return the one in which per class RATIO most balancing.
    Don't support RGB label now.
    Args:
        label: 1. [HW]; 2. [HW1]; 3. one_hot-[HWC]; 4. rgb-[HWC]-目前不支持rbg!!!
            Note: 1&2 class_num need to be specified!!!
        num_classes: default=0, only one_hot label don't have to specify.
    Returns:
        image: croped image
        label: croped laebl
    '''
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_w < image.shape[1]) and (crop_h < image.shape[0]):
        x, y = [0, 0, 0, 0], [0, 0, 0, 0]
        rate = [1., 1., 1., 1.]  # store min RATIO
        sum_area = crop_h * crop_w

        if len(label.shape) == 2 or label.shape[-1] == 1:  # [HW] or [HW1]
            for i in range(4):  # try four times and choose the max rate's x,y
                x[i] = np.random.randint(0, image.shape[1] - crop_w)  # W
                y[i] = np.random.randint(0, image.shape[0] - crop_h)  # H
                tmp_label = label[y[i]:y[i] + crop_h, x[i]:x[i] + crop_w].crop()

                for j in range(num_classes):
                    indArr = [tmp_label == j]
                    tmp_rate = np.sum(indArr) / sum_area
                    if tmp_rate == 0:
                        rate[i] = tmp_rate
                        break
                    elif tmp_rate < rate[i]:
                        rate[i] = tmp_rate

        else:  # [HWC] - only support one_hot label now
            for i in range(4):
                x[i] = np.random.randint(0, image.shape[1] - crop_w)  # W
                y[i] = np.random.randint(0, image.shape[0] - crop_h)  # H
                tmp_label = label[y[i]:y[i] + crop_h, x[i]:x[i] + crop_w, :].crop()

                # For one_hot label
                for j in range(tmp_label.shape[2]):  # traverse all channel-class
                    indArr = tmp_label[:, :, j]
                    tmp_rate = np.sum(indArr) / sum_area
                    if tmp_rate == 0:
                        rate[i] = tmp_rate
                        break
                    elif tmp_rate < rate[i]:
                        rate[i] = tmp_rate

        ind = rate.index(max(rate))  # choose the max RATIO area
        x, y = x[ind], y[ind]

        try:
            label = label[y:y + crop_h, x:x + crop_w]
        except Exception:
            label = label[y:y + crop_h, x:x + crop_w, :]

        image = image[y:y + crop_h, x:x + crop_w, :]
        return image, label

    elif (crop_w == image.shape[1] and crop_h == image.shape[0]):
        return image, label

    else:
        raise Exception('Crop shape exceeds image dimensions!')


def resize_img(image, new_size=[256]):
    '''
    Resize the image into a new shape (keep aspect ratio,
    let the shortest side equal to the target heigth/width,
    then crop it into the targe height and width).
    Args:
        new_size - a tuple may have only one element(H=W) or two elements(H, W).
    '''
    if len(new_size) == 2:
        H, W = new_size
        image = cv2.resize(image, (W, H))
    elif len(new_size) == 1:
        h, w = image.shape[:2]
        if w > h:  # if image width > image height
            H, W = new_size[0], int(w*new_size[0]/h)
            st = int((W-H)/2)
            image = cv2.resize(image, (W, H))[:, st:st+H]
        else:  # if image height > image width
            H, W = int(w*new_size[0]/h), new_size[0]
            st = int((H-W)/2)
            image = cv2.resize(image, (W, H))[st:st+H, :]
    else:
        ValueError('Incorrect new_size!')

    return image


def unify_imgsize(img_dir, size=(256, 256), interpolation=cv2.INTER_NEAREST):
    '''
    Unify image size.
    Args:
        size: Uniform size(must be tuple)
        interpolation: Interpolation method of zoom image
    '''
    num = 1
    image_names = sorted(os.listdir(img_dir))
    for name in tqdm(image_names):
        img = cv2.imread(img_dir+'/'+name, -1)
        if img.shape[:2] != size:
            img = cv2.resize(img, size[::-1], interpolation)
        cv2.imwrite(img_dir+'/'+name, img, [1, 100])
        num += 1


def label_pretreat(label_dir, label_values):
    '''
    Pre-treat the orignal RGB label, to fix the apparent bug.
    (Actually it eliminate the wrong colors that are not in class_dict)
    By the way, unify the dtype of label imgs to png.
    '''
    l_names = filelist(label_dir, ifPath=True)
    for name in tqdm(l_names):
        label = cv2.imread(name, 1)  # read in RGB
        os.remove(name)
        name = os.path.splitext(name)[0] + '.png'
        new_label = np.zeros(label.shape, label.dtype)
        # Fix the color(stand for a class) by class-info
        for color in label_values:
            equality = np.equal(label, color)
            ind_mat = np.all(equality, axis=-1)
            new_label[ind_mat] = color  # this color list can be customized
        cv2.imwrite(name, new_label)  # new_label-BGR(according to class_dict)


# **********************************************
# ********** Commen data augment ***************
# **********************************************
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    '''rotate xb, yb'''
    h, w = xb.shape[0], xb.shape[1]
    M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (w, h))
    yb = cv2.warpAffine(yb, M_rotate, (w, h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(image, gamma=0.01):
    ''' 添加点噪声 '''
    num = image.shape[0] * image.shape[1] * gamma
    for i in range(num):  # 100 - control the num of piexls to add noise
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        image[temp_x][temp_y] = np.random.randint(0, 255)

    return image


def data_augment(image, label=None, crop_size=None, zoom_size=None, if_flip=True, if_roate=True):
    '''
    Simple data augment function to random crop,flip and roate image and label.
    Args:
        image: array of image, [HWC] or [HW1] or [HW].
        label: array of label, [HWC] or [HW1] or [HW].
        crop_size: [h, w]
        zoom_size: [h, w]
        if_flip: if flip the image(and label), defualt = True
        if_roate: if roate the image(and label), defualt = True

    Returns:
        image, label: processed image and label
    '''
    if label is None:
        if crop_size is not None:
            image = random_crop(image, crop_size[0], crop_size[1])
        if zoom_size is not None:
            image = cv2.resize(
                image, (zoom_size[1], zoom_size[0]), interpolation=cv2.INTER_LINEAR)
        # Flip
        if if_flip and np.random.randint(0, 2):
            image = cv2.flip(image, 1)
        if if_flip and np.random.randint(0, 2):
            image = cv2.flip(image, 0)
        # Roate
        iflag = np.random.randint(0, 2)
        if if_roate and iflag:
            angle_table = [90, 180, 270]
            angle = angle_table[np.random.randint(0, 3)]
            h, w = image.shape[0], image.shape[1]
            M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            image = cv2.warpAffine(image, M_rotate, (w, h))
        return image
    else:
        if crop_size is not None:
            image, label = random_crop_2(image, label, crop_size[0], crop_size[1])
        if zoom_size is not None:
            image = cv2.resize(image, (zoom_size[1], zoom_size[0]), cv2.INTER_LINEAR)
            label = cv2.resize(label, (zoom_size[1], zoom_size[0]), cv2.INTER_NEAREST)
        # Flip
        if if_flip and np.random.randint(0, 2):
            image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        if if_flip and np.random.randint(0, 2):
            image, label = cv2.flip(image, 0), cv2.flip(label, 0)
        # Roate
        iflag = np.random.randint(0, 2)
        if if_roate and iflag:
            angle_table = [90, 180, 270]
            angle = angle_table[np.random.randint(0, 3)]
            image, label = rotate(image, label, angle)
        return image, label


# **********************************************
# ************** HDF5 tools ********************
# **********************************************
# def label2hdf5(input_dir, output_path, label_values=None, class_num=None):
#     """
#     Transform a label image into one-hot format (depth is num_classes),
#     and save them all in a dataset(key:"labels") of a h5 flie.

#     Args:
#         input_dir: The dir of label images [HW] or [HWC](BGR)
#         output_path: The whole path of the output h5 flie.
#         label_values: A dictionairy of class--> pixel values, defualt=None
#         class_num: Num of classes, defualt=None
#             Note:
#                 1. If class_num is specified, input labels is [HW](imread in gray mode),
#                     call one_hot_1
#                 2. If label_values is specified, input labels is [HWC]BGR, call one_hot_2;

#     Returns
#         None, but will create a 4d dataset store all one hot labels(in a h5 file).

#     Notice:
#         the dtype of the dataset is uint8(np.uint8).
#         the key of labels dataset is "label".

#     """
#     print("*******Execute label2hdf5()*******")
#     input_names = sorted(os.listdir(input_dir))  # only the name of labels
#     # Carefull!! Since the os.listdir() return a list in atribute order, sort it

#     # Read first of them to get shape
#     tmp_input = cv2.imread(input_dir + "/" + input_names[0], -1)
#     size = [0, 0, 0, 0]  # NHWC
#     size[0] = len(input_names)  # get N
#     size[1], size[2] = tmp_input.shape[0], tmp_input.shape[1]  # get H,W

#     if class_num is not None:  # [HW], class_num must not None
#         size[3] = class_num
#     elif label_values is not None:  # [HWC]-BGR
#         size[3] = size[3] = len(label_values)
#     else:
#         ValueError("label_values or class_num must specify one.")

#     # Open(or create) a hdf5 flie to save all the labels
#     f = h5py.File(output_path, 'a')
#     dataset = f.create_dataset("label", shape=size, dtype=np.uint8)  # create a 4d dataset
#     # Treat per label in the list
#     if class_num is not None:  # [HW], class_num must not None
#         for i in tqdm(range(len(input_names))):
#             label = cv2.imread(input_dir + "/" + input_names[i], 0)
#             dataset[i, :] = one_hot_1(label, class_num)  # return uint8 array
#     elif label_values is not None:  # [HWC]-RGB
#         target = open(os.path.dirname(output_path) + "/train_gt.txt", 'w')
#         for i in tqdm(range(len(input_names))):
#             label = cv2.imread(input_dir + "/" + input_names[i], -1)
#             dataset[i, :] = one_hot_2(label, label_values)  # return uint8 array
#             target.write(input_names[i]+'\n')
#     print(dataset.shape, dataset.dtype)
#     print("*******Finish label2hdf5()*******")
#     f.close()


# def img2hdf5(input_dir, output_path):
#     """
#     Save images and their names into h5 file

#     Args:
#         input_dir: The dir of images.
#         output_path: The whole path of the output h5 flie.

#     Returns:
#         None.
#         But will create a h5 file with a 4d dataset store all images(RGB),
#             and a 1d dataset store all their names.
#     Notice:
#         the dtype of the dataset is uint8(np.uint8).
#         the key of image dataset is "image".(RGB)
#         the key of image name dataset is "name"

#     """
#     print("*******Execute img2hdf5()*******")
#     input_names = sorted(os.listdir(input_dir))  # only the name of images
#     # Carefull!! Since the os.listdir() return a list in atribute order, sort it
#     names = []  # 将文件名转utf8编码，才能存入hdf5的dataset中
#     for j in input_names:
#         names.append(j.encode('utf8'))

#     # Read first of them to get shape
#     tmp_input = load_image(input_dir + "/" + input_names[0], 1)
#     size = [0, 0, 0, 0]  # NHWC
#     size[0] = len(input_names)  # get N
#     size[1:] = tmp_input.shape  # get HWC

#     # Open(or create) a hdf5 flie to save all the images and their names
#     f = h5py.File(output_path, 'w')
#     f.create_dataset("name", data=names)  # 创建name数据集存储images 的utf8编码的文件名
#     # Create an empty 4d dataset(image) to store images
#     dataset = f.create_dataset("image", shape=size, dtype=np.uint8)
#     # Treat per img in the list
#     target = open(os.path.dirname(output_path) + "/train.txt", 'w')
#     for i in tqdm(range(len(input_names))):
#         input_path = input_dir + "/" + input_names[i]  # the whole path
#         img = load_image(input_path, 1)
#         # Extension pocessing
#         # for c in range(img.shape[-1]):
#         #     img[:, :, c] = cv2.equalizeHist(img[:, :, c])
#         dataset[i, :] = img
#         target.write(input_names[i]+'\n')

#     print(dataset.shape, dataset.dtype)
#     print("*******Finish img2hdf5()*******")
#     f.close()


# def img2hdf5_2(input_dir, output_path, newsize=None):
#     """
#     For classified, save images and their names into h5 file,
#     meanwhile make their labels[NC].

#     Args:
#         input_dir: The dir of images(in several folders).
#         output_path: The whole path of the output h5 flie.

#     Returns:
#         None.
#         But will create a h5 file with a 4d dataset store all images(RGB),
#             and a 1d dataset store all their names.
#     Notice:
#         the dtype of the dataset is uint8(np.uint8).(RGB)
#         the key of image dataset is "image".
#         the key of label dataset is "label".
#         the key of image name dataset is "name"
#         the key of class name dataset is "class"

#     """
#     print("*******Execute img2hdf5()*******")
#     # Fisrt travals get total image num and all class names and image names
#     image_names = []  # all image names
#     folder_names = sorted(os.listdir(input_dir))  # also name of classes
#     for i, f in enumerate(folder_names):
#         file_names = sorted(os.listdir(input_dir+'/'+f))  # name of images
#         image_names += file_names

#     utf_names = [name.encode('utf8') for name in image_names]  # 转utf8存入
#     utf_classNames = [name.encode('utf8') for name in folder_names]

#     # Read first of them to get shape
#     tmp_input = load_image(os.path.join(input_dir, folder_names[0], image_names[0]), 1)
#     img_size = [0, 0, 0, 0]  # NHWC(channel), size of image dataset
#     label_size = [0, 0]  # NC(classnum), size of label dataset
#     img_size[0] = label_size[0] = len(image_names)  # get N
#     img_size[1:] = tmp_input.shape  # get HWC si.extend(ze
#     label_size[1] = len(folder_names)  # get class num

#     # Open(or create) a hdf5 flie to save all the images and their names
#     f = h5py.File(output_path, 'w')
#     f.create_dataset("name", data=utf_names)  # 创建name数据集存class_dict储images 的utf8编码的文件名
#     f.create_dataset('class', data=utf_classNames)
#     # Create an empty 4d dataset(image) to store images
#     imageSet = f.create_dataset("image", shape=img_size, dtype=np.uint8)
#     labelSet = f.create_dataset("label", shape=label_size, dtype=np.uint8)
#     # Save per img.extend(
#     i = 0
#     for c, fo in tqdm(enumerate(folder_names)):
#         file_names = sorted(os.listdir(input_dir+'/'+fo))
#         for fi in file_names:
#             img = load_image(os.path.join(input_dir, fo, fi), 1)
#             if newsize is not None:
#                 img = cv2.resize(img, (newsize[1], newsize[0]))
#             imageSet[i, :] = img
#             labelSet[i, c] = 1
#             i += 1

#     # print(imageSet.shape, imageSet.dtype)
#     print("*******Finish img2hdf5()*******")
#     f.close()


# # **********************************************
# # ************ Main functions ******************
# # **********************************************
# def main_of_hdf5():
#     ''' Cover images to hdf5 format. '''

#     dataset_dir = '/home/tao/Data/small'
#     img_dir = dataset_dir + '/train_seg'
#     label_dir = dataset_dir + '/train_labels'
#     class_dict_path = dataset_dir + '/class_dict.txt'
#     h5_path = dataset_dir + '/trainData.h5'

#     _, class_values = get_label_info(class_dict_path)
#     img2hdf5(img_dir, h5_path)
#     label2hdf5(label_dir, h5_path, label_values=class_values)


# def crop_imgs(image_dir, out_dir=None, crop_params=None):
#     ''' Slide crop images into small piece and save them. '''
#     if out_dir is None:
#         out_dir = image_dir

#     image_names = sorted(os.listdir(image_dir))
#     for name in tqdm(image_names):
#         image = cv2.imread(image_dir+'/'+name, -1)
#         # 附加操作
#         # h, w = image.shape[:2]
#         # image = cv2.resize(image, (int(w*0.7), int(h*0.7)))
#         imgs, _ = slide_crop(image, crop_params)
#         for i in range(len(imgs)):
#             save_name = rename_file(name, addstr="_%d" % (i+1))
#             cv2.imwrite(out_dir+'/'+save_name, imgs[i], [1, 100])


# def main_of_img_ops(image_dir, out_dir=None, operations=None):
#     # class_dict_path = '/home/tao/Data/Road/class_dict.txt'
#     # _, class_values = get_label_info(class_dict_path)

#     ext = None
#     read_flag = -1

#     if out_dir is None:
#         out_dir = image_dir
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     imgs_names = filelist(image_dir, extension=ext)
#     for img in tqdm(imgs_names):
#         image = cv2.imread(image_dir+'/'+img, read_flag)
#         if operations is not None:
#             for opera in operations:
#                 image = opera(image)
#         else:
#             # image = class_label(image, class_values)
#             image[image == 255] = 1
#             # image = colour_code_label(image, class_values)
#             pass
#         cv2.imwrite(out_dir+'/'+img, image, [1, 100])


# def main_of_imgAndlabel_ops(image_dir=None, label_dir=None):
#     # class_dict_path = '/home/tao/Data/Road/class_dict.txt'
#     # _, class_values = get_label_info(class_dict_path)
#     image_dir = '/home/tao/Data/RBDD/train'
#     label_dir = '/home/tao/Data/RBDD/train_labels'
#     img_names = filelist(image_dir)
#     lbl_names = filelist(label_dir)
#     # root = ''
#     # img_names = ['1.jpg', '5.jpg']
#     # lbl_names = ['road_1_pre.png', 'road_5_pre.png']

#     for (img_name, lbl_name) in tqdm(zip(img_names, lbl_names)):
#         image = cv2.imread(image_dir+'/'+img_name, -1)
#         label = cv2.imread(label_dir+'/'+lbl_name, -1)
#         assert image.shape[:2] == label.shape[:2]
#         # Operatrions
#         cv2.imwrite(label_dir+'/'+lbl_name, label)

#         # image = mask_img(image, label, [255, 0, 0])
#         # cv2.imwrite(image_dir+'/'+img_name, image, [1, 100])


def main():
    '''
    main().
    '''
    # class_dict_path = '/home/tao/Data/Road/class_dict.txt'
    # _, class_values = get_label_info(class_dict_path)

    # label_dir = '/home/tao/Data/cvpr_road/256/train_labels'
    pass


if __name__ == '__main__':
    # main()
    # main_of_imgAndlabel_ops()
    # in_dir = '/home/tao/Data/cvpr_road/1024/filtered_img/val_labels'
    # out_dir = '/home/tao/Data/cvpr_road/512/val_labels'
    in_dir = '/home/tao/Data/RBDD/BackUp/Orignal_data/val_labels'
    out_dir = '/home/tao/Data/RBDD/512/val_labels'
    # main_of_img_ops(in_dir, out_dir)
    # convert_image_type(in_dir, 'png')

    crop_imgs(in_dir, out_dir, [512, 512, 512])

    pass

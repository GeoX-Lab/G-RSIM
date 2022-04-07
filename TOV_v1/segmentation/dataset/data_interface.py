import os
# import random
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as transform_lib
from albumentations.pytorch.transforms import ToTensorV2

SPECIFIC_DATASET_MODE = {
}


def get_data_dir(ds, data_dir):
    if type(data_dir) == str:
        return data_dir
    elif type(data_dir) == dict:
        return data_dir[ds]


def datasets_init(ds, data_dir, class_dict, dtype='RGB', full_train=False, train_init_data=None, **kwargs):
    """ Repackaged the function `segdataset_init()`.
    """
    data_mode = 0
    if ds in SPECIFIC_DATASET_MODE.keys():
        data_mode = SPECIFIC_DATASET_MODE[ds]

    root = get_data_dir(ds, data_dir)

    init_data = segdataset_init(root, dtype, full_train, data_mode)
    init_data['class_dict'] = class_dict

    if train_init_data:
        with open(train_init_data, 'rb') as f:
            ext_train_init_data = pkl.load(f)
        init_data['train'] = ext_train_init_data['train']
        print(f'Use {train_init_data} for training')
    return init_data


class DInterface(pl.LightningDataModule):

    def __init__(self, dataset='',
                 normalize=True,
                 data_module='',
                 data_class='SegData',
                 **kwargs):
        super().__init__()
        self.dataset = dataset
        self.kwargs = kwargs
        self.data_module = data_module
        self.data_class = data_class

        self.normalize = normalize
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.init_data = datasets_init(dataset, **kwargs)
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_transforms = self.default_transforms(train=True)
            val_transforms = self.default_transforms()

            self.trainset = self.instancialize(split='train',
                                               ratio=self.kwargs['train_scale'],
                                               transforms=train_transforms)
            self.valset = self.instancialize(split='val',
                                             ratio=self.kwargs['val_scale'],
                                             transforms=val_transforms)

        # Assign train/val datasets for use in dataloaders
        if stage == 'validate' or stage is None:
            val_transforms = self.default_transforms()
            self.valset = self.instancialize(
                split='val', ratio=1, transforms=val_transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_transforms = self.default_transforms()
            self.testset = self.instancialize(
                split='test', ratio=1, transforms=test_transforms)

    @property
    def num_samples(self) -> int:
        return self.get_num_samples()

    @property
    def num_classes(self) -> int:
        return len(self.init_data['class_dict'])

    def get_num_samples(self, split='train', full_train=False) -> int:
        ratio = self.kwargs['train_scale'] if split == 'train' else 1

        total_num = len(self.init_data[split]['image'][self.kwargs['dtype'][0]])
        if ratio <= 1:
            use_num = max(round(total_num*ratio), 1)
        else:
            use_num = min(int(ratio), total_num)

        return use_num

    def default_transforms(self, train=False):
        df_transforms = []
        if train:
            if self.train_transforms is not None:
                return self.train_transforms

            df_transforms.extend([
                transform_lib.RandomResizedCrop(
                    *self.kwargs['input_size'], scale=(.6, 1.)),
                # RandomRotation([0, 90, 180, 270]),
                transform_lib.HorizontalFlip(),
                transform_lib.VerticalFlip(),
            ])
        else:
            if self.val_transforms is not None:
                return self.val_transforms

            df_transforms.append(transform_lib.Resize(*self.kwargs['input_size']))

        # Basic
        if self.normalize:
            df_transforms.append(self.get_normalizer())
        df_transforms.append(ToTensorV2())  # TODO: support no label
        return transform_lib.Compose(df_transforms)

    def get_normalizer(self):
        return transform_lib.Normalize(self.kwargs['mean'], self.kwargs['std'])

    def train_dataloader(self):
        prefetch_num = 2
        if self.num_workers and self.kwargs['super_prefetch']:
            prefetch_num = self.batch_size//self.num_workers
        # print('drop_last is {}'.format(self.kwargs['mode_name'] == 'pretrain'))
        return DataLoader(
            self.trainset, self.batch_size, True,
            num_workers=self.num_workers,
            pin_memory=(self.kwargs['num_gpus'] > 0),
            drop_last=True if self.kwargs['mode_name'] == 'pretrain' else False,
            prefetch_factor=prefetch_num  # prefetch the next batch data
        )

    def val_dataloader(self):
        prefetch_num = 2
        if self.num_workers and self.kwargs['super_prefetch']:
            prefetch_num = self.batch_size//self.num_workers
        return DataLoader(
            self.valset, self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.kwargs['num_gpus'] > 0),
            prefetch_factor=prefetch_num)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size, num_workers=self.num_workers)

    def load_data_module(self):
        if self.dataset in ['']:  # Auto set data module and class for specific dataset
            raise ValueError(f'Invalid Dataset {self.dataset}')
            self.data_module = ''
            self.data_class = ''

        try:
            self.data_module = getattr(importlib.import_module(
                '.'+self.data_module, package=__package__), self.data_class)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f'Invalid Module File Name {self.data_module}!')
        except AttributeError:
            raise AttributeError(f'Invalid Class Name {self.data_class}!')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(self.init_data, **args1)


def segdataset_init(root, dtype='RGB', full_train=False, mode=0, seed=0):
    '''
    Initilize the segmentation dataset dict:
        {'train': {'image': {'RGB': [img_pths], 'SAR': [img_pths], ...},
                   'label': [lbl_pths]},
         'val': {'image': {'RGB': [img_pths], 'SAR': [img_pths]},
                 'label': [lbl_pths]},
        }
    Args:
        root - The dir of the train(test) dataset folder.
        dtype - The type of data, ['RGB', 'SAR']
        mode - 文件的组织结构不同
            0: 预测划分了train/val/test but only RGB data
            1: 预测划分了train/val/test and have multi-datatype
        full_train - wether return all the images for train
    '''
    init_dataset = {k: {'image': {}, 'label': []} for k in ['train', 'val', 'test']}
    lbl_dir_suffix = ''
    for suffix in ['_lbl', '_label', '_labels']:
        for sp in ['train', 'val', 'test']:
            test_pth = root + '/' + sp + suffix
            if os.path.exists(test_pth):
                lbl_dir_suffix = suffix
                break

    if mode == 0:  # only RGB data
        for sp in ['train', 'val', 'test']:
            lbl_dir = root + '/' + sp + lbl_dir_suffix
            if os.path.exists(lbl_dir):
                lbl_pths = [lbl_dir+'/'+name for name in sorted(os.listdir(lbl_dir))]
                init_dataset[sp]['label'] = lbl_pths
                if full_train and sp != 'train':
                    init_dataset['train']['label'] += lbl_pths
            else:
                print('No %sing set label found at: %s.' % (sp, lbl_dir))
            img_dir = root + '/' + sp
            if os.path.exists(img_dir):
                img_pths = [img_dir+'/'+name for name in sorted(os.listdir(img_dir))]
                init_dataset[sp]['image']['RGB'] = img_pths
                if full_train and sp != 'train':
                    init_dataset['train']['image']['RGB'] += img_pths
            else:
                print('No %sing set image found at: %s.' % (sp, img_dir))

    elif mode == 1:
        for sp in ['train', 'val', 'test']:
            lbl_dir = root + '/' + sp + lbl_dir_suffix
            if os.path.exists(lbl_dir):
                lbl_pths = [lbl_dir+'/'+name for name in sorted(os.listdir(lbl_dir))]
                init_dataset[sp]['label'] = lbl_pths
                if full_train and sp != 'train':
                    init_dataset['train']['label'] += lbl_pths
            else:
                print('No %sing set label found at: %s.' % (sp, lbl_dir))
            for dt in dtype:
                img_dir = root + '/%s_%s' % (sp, dt)
                if os.path.exists(img_dir):
                    img_pths = [img_dir+'/'+name for name in sorted(os.listdir(img_dir))]
                    init_dataset[sp]['image'][dt] = img_pths
                    if full_train and sp != 'train':
                        init_dataset['train']['image'][dt] += img_pths
                else:
                    print('No %sing set image type of %s found at: %s.' % (sp, dt, img_dir))
    return init_dataset

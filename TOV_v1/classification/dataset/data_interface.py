import os
from typing import Dict
import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib

SPECIFIC_DATASET_MODE = {
    'rsd46': 2
}


def get_data_dir(ds, data_dir):
    if type(data_dir) == str:
        return data_dir
    elif type(data_dir) == dict:
        return data_dir[ds]


def datasets_init(ds, data_dir, train_val_ratio, full_train=False, **kwargs):
    """ Repackaged the function `classifydataset_init()` to aggregate multiple datasets.
    """
    data_mode = 1
    if ds in SPECIFIC_DATASET_MODE.keys():
        data_mode = SPECIFIC_DATASET_MODE[ds]

    root = get_data_dir(ds, data_dir)

    init_data = classifydataset_init(
        root, kwargs.get('class_dict', None), train_val_ratio, full_train, data_mode)

    return init_data


class DInterface(pl.LightningDataModule):

    def __init__(self, dataset='',
                 normalize=True,
                 data_module='',
                 data_class='ClassfyData',
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
            # print('*'*20, '\r\nTemp setting: add all `SSD_OSMv1` for SSL\r\n', '*'*20)
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
        return len(self.init_data['train'])

    def get_num_samples(self, split='train', full_train=False) -> int:
        ratio = self.kwargs['train_scale'] if split == 'train' else 1
        total_num = 0
        for samples in self.init_data[split].values():
            if full_train:
                use_num = len(samples)
            else:
                if ratio <= 1:
                    use_num = max(round(len(samples)*ratio), 1)
                else:
                    use_num = min(int(ratio), len(samples))
            total_num += use_num

        return total_num

    def default_transforms(self, train=False):
        df_transforms = []
        if train:
            if self.train_transforms is not None:
                return self.train_transforms

            df_transforms.extend([
                transform_lib.RandomResizedCrop(
                    self.kwargs['input_size'], scale=(.8, 1.)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomVerticalFlip(),
            ])
        else:
            if self.val_transforms is not None:
                return self.val_transforms

            df_transforms.append(transform_lib.Resize(self.kwargs['input_size']))

        # Basic
        df_transforms.append(transform_lib.ToTensor())
        if self.normalize:
            df_transforms.append(self.get_normalizer())
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
            pin_memory=(self.kwargs['gpus'] > 0),
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
            pin_memory=(self.kwargs['gpus'] > 0),
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


def classifydataset_init(root, class_dict=None, ratio=[0.5, 0.3, 0.2], full_train=False, mode=0, seed=0):
    '''
    Initialization of classification data, that is, obtaining all file paths and their labels for the training (test) set
    Args:
        root (string): The dir of the train(test) dataset folder.

        class_dict (dict): {class_name1: ind1, class_name2: ind2, ...}

        ratio - Control the number of training/val/testing samples,
                [train_ratio, val_ratio, test_ratio] or [train_ratio, val_ratio]

        mode - Organizational structure of samples
            1: Each folder put a kind of sample, and training test samples mixed together
            2: The train, val, test set are placed in different folders

        full_train - wether return all the images for train
    '''
    init_dataset = {k: {} for k in ['train', 'val', 'test']}
    # cls_table = category.table

    def _find_classes(data_dir: str) -> Dict[str, int]:
        classes = [fld.name for fld in os.scandir(data_dir) if fld.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    if mode == 1:
        if class_dict is None:
            class_dict = _find_classes(root)
        # loop floders
        for f in sorted(os.listdir(root)):
            cls = class_dict[f]
            img_names = sorted(os.listdir(root + '/' + f))
            samples = [(root+'/'+f+'/'+x, cls) for x in img_names]
            n = len(img_names)
            train_num, val_num = round(n * ratio[0]), round(n * ratio[1])
            if full_train:
                init_dataset['train'].update({f: samples[:]})
            else:
                init_dataset['train'].update({f: samples[: train_num]})
            init_dataset['val'].update({f: samples[train_num: train_num + val_num]})
            init_dataset['test'].update({f: samples[train_num + val_num:]})

    elif mode == 2:
        if class_dict is None:
            for split in ['train', 'val', 'test']:
                data_dir = root + '/' + split
                if os.path.exists(data_dir):
                    class_dict = _find_classes(data_dir)
                    break
        for split in ['train', 'val', 'test']:
            data_dir = root + '/' + split
            if os.path.exists(data_dir):
                for f in sorted(os.listdir(data_dir)):
                    cls = class_dict[f]
                    img_names = sorted(os.listdir(data_dir + '/' + f))
                    samples = [(data_dir+'/'+f+'/'+x, cls) for x in img_names]
                    init_dataset[split].update({f: samples})
                    if full_train and split != 'train':
                        if len(samples) == 0:
                            continue
                        init_dataset['train'][f].extend(samples)
            else:
                print('No %sing set found at: %s.' % (split, data_dir))

    init_dataset['class_dict'] = class_dict
    return init_dataset

import torch
import inspect
import importlib

from torchvision import models as tv_models

from .basic_model import BasicModel


__all__ = ['BasicModel']


def build_model(**hparams):
    name = hparams['model_name']
    # netnum mode
    if name.isdigit():
        net_kwargs = parse_netnum(name)
    else:
        pass
    name = net_kwargs['model']
    # Change the `snake_case.py` file name to `CamelCase` class name.
    # Please always name your model file name as `snake_case.py` and
    # class name corresponding `CamelCase`.
    module_name = name.split('_')[0].lower()

    # * build model *
    backbone = build_torchvision_backbone(hparams, net_kwargs)
    backbone = modify_tv_model(backbone, hparams['in_channel'])

    try:
        Model = getattr(importlib.import_module(
            '.'+module_name, package=__package__), name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f'Invalid Module File Name {module_name}!')
    except AttributeError:
        raise AttributeError(f'Invalid Class Name {name}!')

    net_kwargs['backbone'] = backbone
    model = instancialize(Model, hparams, **net_kwargs)
    return model


def build_torchvision_backbone(hparams, net_kwargs):
    """ Build torchvision backbone.
    """
    pretrained = hparams.get('pretrained', False)

    tv_kwargs = net_kwargs['torchvision_args']

    if 'resnet' in net_kwargs['backbone'] or 'resnext' in net_kwargs['backbone']:
        try:
            tv_target_args = inspect.getargspec(tv_models.ResNet.__init__).args[1:]
        except ValueError:
            # For torchvision > 0.9
            tv_target_args = ['replace_stride_with_dilation']
        for k in tv_target_args:
            if k in hparams.keys() and k != 'num_classes':
                tv_kwargs[k] = hparams[k]

    bb = tv_models.__dict__[net_kwargs['backbone']](
        pretrained=pretrained, **tv_kwargs)
    if pretrained:
        print('****** Load ImageNet pretrained CKPT ******')
    bb.model_name = net_kwargs['backbone']
    return bb


def modify_tv_model(backbone, in_channel=3):
    if in_channel != 3:
        if hasattr(backbone, 'conv1'):  # resnet
            # conv1_size = backbone.conv1.weights.shape
            oc = backbone.conv1
            backbone.conv1 = torch.nn.Conv2d(
                in_channel, oc.out_channels,
                oc.kernel_size, oc.stride, oc.padding,
                bias=(oc.bias is not None))
        elif hasattr(backbone, 'features'):  # vgg,
            oc = backbone.features[0]
            backbone.features[0] = torch.nn.Conv2d(
                in_channel, oc.out_channels,
                oc.kernel_size, oc.stride, oc.padding,
                bias=(oc.bias is not None))

    # loop layers and get last conv channels
    out_dim = None
    for name, m in backbone.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            out_dim = m.out_channels
    backbone.out_dim = out_dim

    # Del fc leayer of backbone
    if hasattr(backbone, 'fc'):
        backbone.fc = None
    elif hasattr(backbone, 'classifier'):
        backbone.classifier = None

    return backbone


def instancialize(Model, hparams, **other_args):
    """ Instancialize a model using the corresponding parameters
        from args dictionary. You can also input any args
        to overwrite the corresponding value in args.
    """
    class_args = inspect.getargspec(Model.__init__).args[1:]
    inkeys = hparams.keys()
    net_kwargs = {}
    for k in class_args:
        if k in inkeys:
            net_kwargs[k] = hparams[k]
        if k in other_args.keys():
            net_kwargs[k] = other_args[k]
    # net_kwargs.update(other_args)
    return Model(**net_kwargs)


def parse_netnum(netnum, **tv_kwargs):
    '''
    Args:
        net_num: the code of net information,
        The 1st to 3rd digits denote the main framework of the model:
            0 - Encoder (for classification ...)
                x0 - Orginal/Standard torchvision model
                x1 - Encoder-Classifier arch: Package torchvision models into `ClsNet`
            1 - EncoderDecoder (for segmentation)
                x0 - Orginal/Standard torchvision model
                x1 - Encoder-Classifier arch: Package torchvision models into `ClsNet`

        The 4th-5th digits denote the architecture of Backbone:
            xxx0 - A temporary space reserved for comparative tests
                xxx00 - No backbone
                xxx01 - Discriminator_GANs
            xxx1 - alexnet
                xxx11 - alexnet
                xxx12 - alexnet_small

            xxx2 - resnet
                xxx21 - resnet18
                xxx22 - resnet34
                xxx23 - resnet50
                xxx24 - resnet101
                xxx25 - resnet18_small
                xxx26 - resnet34_small
            xxx3 - vgg
                xxx31 - vgg11 (vgg11_bn)
                xxx32 - vgg13 (vgg13_bn)
                xxx33 - vgg16 (vgg16_bn)
                xxx34 - vgg19 (vgg19_bn)
            xxx4 - googlenet
                xxx41 - googlenet
            xxx5 - inception
                xxx51 - inception_v2
                xxx52 - inception_v3
                xxx53 - xception

        The 6th digit denotes extra operations for backbone:
            xxxxx0 - None
            xxxxx1 - enhance features: sobel
        The 7th digit denotes extra operations for the whole model:
            xxxxxx0 - None
            xxxxxx1 - PointRend (Segmentation)
            xxxxxx2 - Sync Batchnorm

        The nth digit denotes ...

        The last digit denotes the output_stride(OS) of backbone.
            0 - Org
            1 - 8
            2 - 16
    '''
    assert type(netnum) == str
    paras = [int(n) for n in netnum]  # str -> int
    print('\nNetnum:', netnum)

    def get_backbone(arch_num):
        if arch_num[0] == '0':  # A temporary space reserved for comparative tests
            bb_arch = None
        else:  # Common backbone/classification network
            bb_arch = {
                '11': 'alexnet',
                '12': 'alexnet',  # small

                '21': 'resnet18',
                '22': 'resnet34',
                '23': 'resnet50',
                '24': 'resnet101',
                '25': 'resnet152',
                '26': 'resnext50_32x4d',
                '27': 'resnext101_32x8d',
                '28': 'wide_resnet50_2',
                '29': 'wide_resnet101_2',

                '31': 'resnet18',  # small
                '32': 'resnet34',  # small
                '33': 'resnet50',  # small

                '41': 'vgg11_bn',
                '42': 'vgg13_bn',
                '43': 'vgg16_bn',
                '44': 'vgg19_bn',
                '51': 'googlenet',
                '61': 'inception_v2',
                '62': 'inception_v3',
                '73': 'xception',
            }[arch_num]

            # Specify OS(output_stride) of backbone
            if paras[-1]:
                assert arch_num in ['23', '24', '27']  # TODO: OS in ception
                # replace stride with dilation in resnet
                dilations = {1: [0, 0, 1], 2: [0, 1, 1]}[paras[-1]]
                tv_kwargs.update({'replace_stride_with_dilation': dilations})

        return bb_arch

    net_kwargs = {}
    # * Backbone *
    backbone = get_backbone(netnum[3:5])
    if backbone is not None:
        net_kwargs['backbone'] = backbone
    else:
        raise ValueError(
            'Unrecognize code {} for backbone arch'.format(
                netnum[3:5]))

    # * Extra operations for backbone *
    net_kwargs['enhance_features'] = []
    if paras[5] == 1:
        net_kwargs['enhance_features'].append('sobel')

    # * Package model *
    if paras[0] == 0:
        # * Encoder org *
        if paras[1] == 0:
            net_kwargs['model'] = 'torchvision'
        # * Encoder-Classifier *
        elif paras[1] == 1:  # Package the torchvision models into a unified format
            net_kwargs['model'] = 'cls_net'
        else:
            raise ValueError('Not support net_num (2nd digits)')
    elif paras[0] == 1:
        # * FCN *
        if paras[1] == 0:
            net_kwargs['model'] = {
                0: 'FCN', 1: 'FCN_d8', 2: 'FCN_d16',
            }[paras[2]]
        else:
            raise ValueError('Not support net_num (2nd digits)')
    else:
        raise ValueError('Not support net_num (1st digits)')

    net_kwargs['torchvision_args'] = tv_kwargs
    return net_kwargs

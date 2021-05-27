import platform
import os
import time
# import config as config_lib

TYPE2BAND = {'RGB': 3, 'RGBIR': 4, 'SAR': 1, 'TEN': 10, 'ALL': 12, 'MS': 13}  # 'ALL' for bigearth data; 'MS' for EuroSAT

from pathlib import Path

def unify_type(param, ptype=list, repeat=1):
    ''' Unify the type of param.

    Args:
        ptype: support list or tuple
        repeat: The times of repeating param in a list or tuple type.
    '''
    if repeat == 1:
        if type(param) is not ptype:
            if ptype == list:
                param = [param]
            elif ptype == tuple:
                param = (param)
    elif repeat > 1:
        if type(param) is ptype and len(param) == repeat:
            return param
        elif type(param) is list:
            param = param * repeat
        else:
            param = [param] * repeat
            param = ptype(param)

    return param

def get_config(args=None):
    from .opt_road import DefaultConfig

    if args is not None:
        mOptions = DefaultConfig(args)
    else:
        mOptions = DefaultConfig()

    return mOptions
def get_opt(args=None):
    '''Get options by name, and may use args to update them.'''

    opts = get_config()
    if args is not None:
        # Extra

        # Use ArgumentParser object to update the default configs
        for k, v in args.__dict__.items():
            if v is not None or not hasattr(opts, k):
                setattr(opts, k, v)

    if opts.n_channels==None:
        opts.n_channels=TYPE2BAND[opts.dtype]
    if opts.dataset_dir==None:
        opts.dataset_dir=opts.root
    if opts.class_dict_road==None:
        opts.class_dict_road=opts.dataset_dir+'/class_dict.txt'
    if opts.ckpt == None:
        opts.ckpt = opts.root + '/Model'
        Path(opts.ckpt).mkdir(parents=True, exist_ok=True)
    if opts.outputimage == None:
        opts.outputimage = opts.root + '/OutImage'
        Path(opts.outputimage).mkdir(parents=True, exist_ok=True)

    return opts




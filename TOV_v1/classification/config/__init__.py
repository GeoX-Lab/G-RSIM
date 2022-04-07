import os
import time
import importlib
import sys
import warnings
from pytorch_lightning.utilities import rank_zero_only

warnings.filterwarnings("ignore")

TYPE2BAND = {'RGB': 3, 'NIR': 1, 'SAR': 1, 'TEN': 10, 'ALL': 12, 'MS': 13}  # 'ALL' for sentienl data; 'MS' for EuroSAT
MODE_NAME = {1: 'train', 2: 'val', 3: 'test', 4: 'finetune',  # 5: 'exp',
             5: '', 6: '', 7: '',
             8: '', 9: 'pretrain', 0: 'debug'}


class __redirection__:
    def __init__(self, mode='console', file_path=None):
        assert mode in ['console', 'file', 'both']

        self.mode = mode
        self.buff = ''
        self.__console__ = sys.stdout

        self.file = None
        if file_path is not None and mode != 'console':
            try:
                self.file = open(file_path, "w", buffering=1)
            except OSError:
                print('Fail to open log_file: {}'.format(
                    file_path))

    @rank_zero_only
    def write(self, output_stream):
        self.buff += output_stream
        if self.mode == 'console':
            self.to_console(output_stream)
        elif self.mode == 'file':
            self.to_file(output_stream)
        elif self.mode == 'both':
            self.to_console(output_stream)
            self.to_file(output_stream)

    @rank_zero_only
    def to_console(self, content):
        sys.stdout = self.__console__
        print(content, end='')
        sys.stdout = self

    @rank_zero_only
    def to_file(self, content):
        if self.file is not None:
            sys.stdout = self.file
            print(content, end='')
            sys.stdout = self

    @rank_zero_only
    def all_to_console(self, flush=False):
        sys.stdout = self.__console__
        print(self.buff, end='')
        sys.stdout = self

    @rank_zero_only
    def all_to_file(self, file_path=None, flush=True):
        if file_path is not None:
            self.open(file_path)
        if self.file is not None:
            sys.stdout = self.file
            print(self.buff, end='')
            sys.stdout = self
            # self.file.close()

    @rank_zero_only
    def open(self, file_path):
        try:
            self.file = open(file_path, "w", buffering=1)
        except OSError:
            print('Fail to open log_file: {}'.format(
                file_path))

    @rank_zero_only
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    @rank_zero_only
    def flush(self):
        self.buff = ''

    @rank_zero_only
    def reset(self):
        sys.stdout = self.__console__


def get_opt(name, args=None, redirection=False):
    '''Get options by name and current platform, and may use args to update them.'''

    get_config = importlib.import_module('config.default').get_config
    opts = get_config(name)
    if args is None:
        return opts  # simple mode

    opts = preprocess_settings(opts, args)

    # Normalize the form of some parameters
    opts.dtype = unify_type(opts.dtype, list)
    opts.input_size = unify_type(opts.input_size, tuple, 2)
    for dt in opts.dtype:
        opts.in_channel += TYPE2BAND[dt]
    opts.mean = opts.mean if len(opts.mean) == opts.in_channel else opts.mean * opts.in_channel
    opts.std = opts.std if len(opts.std) == opts.in_channel else opts.std * opts.in_channel

    # Generate logging flags / info.
    opts.timestamp = time.strftime('%y%j%H%M%S', time.localtime(time.time()))
    mode_digits = len(str(opts.mode))
    opts.mode_name = MODE_NAME[opts.mode // (10**(mode_digits-1))]
    if not hasattr(opts, 'net_suffix'):
        opts.net_suffix = opts.timestamp + '_' + opts.mode_name
    if opts.mode_name == 'finetune':
        assert opts.load_pretrained, '`load_pretrained` must be provided for finetune'
        expnum = 'temp'
        for s in opts.load_pretrained.split('_'):
            if len(s) == 11 and s.isdigit():
                expnum = s
                break
            elif s in ['other']:
                expnum = s
                break
        opts.net_suffix = expnum + '_' + opts.net_suffix

    opts.exp_name = '{}_{}_{}'.format(opts.env, opts.model_name, opts.net_suffix)

    # Re-direction the print
    if redirection:
        print_mode = 'both'
        stream = __redirection__(print_mode)
        sys.stdout = stream

    print('Current Main Mode: {} - {}\n'.format(opts.mode, opts.exp_name))

    # Basic prepare
    if opts.ckpt:
        if not os.path.exists(opts.ckpt):
            os.mkdir(opts.ckpt)

    if redirection:
        return opts, stream
    return opts


def preprocess_settings(opts, args=None):

    # Combine user args with predefined opts
    if args is not None:
        for attr in opts.__dir__():
            if attr.startswith('__') or attr.endswith('__'):
                continue
            value = getattr(opts, attr)
            if attr not in args or args.__dict__[attr] is None:
                setattr(args, attr, value)
        opts = args

    # Add some default settings
    if opts is not None:
        if not hasattr(opts, 'pj') or not opts.pj:
            opts.pj = opts.env

        if opts.max_epochs is None:
            opts.max_epochs = 3

        # Other
        if not hasattr(opts, 'full_train') or opts.full_train is None:
            opts.full_train = False

    return opts


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


def args_list2dict(params):
    if type(params) is dict:
        return params
    elif type(params) is list:
        assert len(params) % 2 == 0, 'Must be paired args'
        options = {}
        for i in range(0, len(params), 2):
            options[params[i]] = params[i+1]

        return options

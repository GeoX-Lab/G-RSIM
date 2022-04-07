
from .tools import (
    load_pretrain_path_by_args, load_model_path_by_args,
    load_ckpt,
    checkpoint_dict_mapping, checkpoint_standardize,
)
from .mycallbacks import (
    MyLogger, MyDebugCallback, MySpeedUpCallback,
    get_checkpoint_callback
)

__all__ = [
    'MyLogger', 'MyDebugCallback', 'MySpeedUpCallback',
    'get_checkpoint_callback',
    'load_pretrain_path_by_args', 'load_model_path_by_args',
    'load_ckpt',
    'checkpoint_dict_mapping', 'checkpoint_standardize',
]


'''
def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the
        first three args.
    Args:
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(
                Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root == version == v_num is None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [
            i for i in list(Path(root).iterdir()) if i.stem.startswith('best')
        ]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir,
                           version=args.load_ver,
                           v_num=args.load_v_num)

class MyTimeLogCallback(pl.Callback):

    def __init__(self, log_file=None):
        super().__init__()
        self.tics = {'total_epoch': 0}
        self.epoch_count = 0
        self.log_file = log_file

    def on_train_start(self, trainer, pl_module):
        if self.log_file is not None:
            try:
                self.log_file = open(self.log_file, 'w')
            except OSError:
                print('Fail to open log_file: {}'.format(
                    self.log_file))
                self.log_file = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.tics['epoch'] = time.time()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        self.epoch_count += 1
        epoch_time = time.time()-self.tics['epoch']
        m, s = divmod(epoch_time, 60)
        h, m = divmod(m, 60)
        print('epoch time = {:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(h, m, s),
              file=self.log_file)
        self.tics['total_epoch'] += epoch_time

    def on_train_end(self, trainer, pl_module):
        avg_epoch_time = self.tics['total_epoch'] / self.epoch_count
        m, s = divmod(avg_epoch_time, 60)
        h, m = divmod(m, 60)
        print('avg epoch time = {:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(h, m, s),
              file=self.log_file)
        if self.log_file:
            self.log_file.close()
'''

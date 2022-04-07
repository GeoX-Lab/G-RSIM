import datetime
import os
import threading
import time

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc


# **************************************************
# ****************** Log Callbacks *****************
# **************************************************
class MyLogger(pl.Callback):
    """ My custom logger for time cost and training/val metrics.
    """
    def __init__(
        self,
        log_dir=None,
        log_name='exp_log',
        epoch_interval: float = 0.2,  # 5 times during a epoch
    ):
        super().__init__()
        self.tics = {'train_total_time': 0, 'val_total_time': 0,
                     'train_epoch_time': 0, 'val_epoch_time': 0}

        self.log_file = None
        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_file = os.path.join(log_dir, log_name+'.log')

        self.epoch_interval = epoch_interval

    # @pl.utilities.rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # if not trainer.is_global_zero:
        #     return
        self.tics['fit_st'] = time.time()
        if self.log_file is not None:
            try:
                self.log_file = open(self.log_file, 'w')
            except OSError:
                print('Fail to open log_file: {}'.format(
                    self.log_file))
                self.log_file = None

    # @pl.utilities.rank_zero_only
    def on_train_start(self, trainer, pl_module):
        # if not trainer.is_global_zero:
        #     return
        self.print('Start training...')
        self.log_n_train_steps = max(round(trainer.num_training_batches * self.epoch_interval), 1)

    # @pl.utilities.rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        # if not trainer.is_global_zero:
        #     return
        self.tics['train_epoch_st'] = time.time()

    # @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not trainer.is_global_zero:
        #     return
        if batch_idx % self.log_n_train_steps == 0:
            time_stamp = datetime.datetime.now().strftime("[%d %H:%M:%S]")
            # loss = outputs['loss'].item()
            # acc = 0
            # if hasattr(pl_module, 'train_acc'):
            #     acc = pl_module.train_acc.compute().item()

            train_batch_log = f'\t{time_stamp} '
            train_batch_log += f'Epoch: [{trainer.current_epoch} | {trainer.max_epochs}] iters: [{batch_idx:5d}]'
            for k, v in trainer.logged_metrics.items():
                # if 'online' in k:
                #     continue
                if ('loss' in k or 'train' in k) and ('epoch' not in k):
                    train_batch_log += ' {}: {:.3f}'.format(k, v.item())
            self.print(train_batch_log)

    # @pl.utilities.rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # if not trainer.is_global_zero:
        #     return
        self.tics['train_epoch_time'] = time.time() - self.tics['train_epoch_st']
        self.tics['train_total_time'] += self.tics['train_epoch_time']
        epoch_time = self.tics['train_epoch_time'] + self.tics['val_epoch_time']

        train_epoch_log = '\tEpoch_time={:.1f}s'.format(epoch_time)
        for k, v in trainer.logged_metrics.items():
            if 'online' in k:
                continue
            if ('loss' in k or 'train' in k) and ('epoch' in k):
                train_epoch_log += ' {}: {:.3f}'.format(k, v.item())
        self.print(train_epoch_log)

    # @pl.utilities.rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        # if not trainer.is_global_zero:
        #     return
        self.tics['val_epoch_st'] = time.time()

    # @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module, unused=None):
        # if not trainer.is_global_zero:
        #     return
        self.tics['val_epoch_time'] = time.time() - self.tics['val_epoch_st']
        self.tics['val_total_time'] += self.tics['val_epoch_time']
        # epoch_time = self.tics['train_epoch_time'] + self.tics['val_epoch_time']

        val_epoch_log = '\t'
        if hasattr(pl_module, 'epoch_scores'):
            for k, v in pl_module.epoch_scores.items():
                if type(v) is np.float64:
                    val_epoch_log += f'val_{k}={v:.2f} '
        else:
            for k, v in trainer.logged_metrics.items():
                if 'online' in k:
                    continue
                if ('val' in k) and ('loss' not in k) and ('train' not in k):
                    val_epoch_log += ' {}: {:.3f}'.format(k, v.item())
        # log_str += f' Epoch_time={epoch_time:.1f}s'
        self.print(val_epoch_log)

    # def on_train_end(self, trainer, pl_module):
    #     pass

    # @pl.utilities.rank_zero_only
    # def on_fit_end(self, trainer, pl_module):
    def on_train_end(self, trainer, pl_module):
        # if not trainer.is_global_zero:
        #     return
        avg_epoch_time = (time.time() - self.tics['fit_st']) / max(trainer.current_epoch, 1)
        m, s = divmod(avg_epoch_time, 60)
        h, m = divmod(m, 60)

        log_str = 'Finish training! Stop at epoch {} (max epoch={}), '.format(
            trainer.current_epoch, trainer.max_epochs)
        log_str += 'avg epoch time={:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(h, m, s)
        self.print(log_str)

        if self.log_file:
            self.log_file.close()

    @pl.utilities.rank_zero_only
    def print(self, *args, **kwargs) -> None:
        log_file = kwargs.pop('file', self.log_file)
        print(*args, file=log_file, **kwargs)


class MyDebugCallback(pl.Callback):

    def on_train_start(self, trainer, pl_module):
        print('on_train_start')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print('on_train_batch_end')

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        print('\non_train_epoch_end\n')

    def on_train_end(self, trainer, pl_module):
        print('on_train_start')


class ThreadKiller(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


# **************************************************
# *************** Get PL callbacks *****************
# **************************************************
def get_checkpoint_callback(dirpath,
                            monitor='train/loss',
                            mode='min',
                            filename="{epoch}",
                            save_last=True,
                            save_top_k=2,
                            every_n_train_steps=None):
    ckpt_callback = plc.ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,  # ckpt_name + "_{epoch}",
        monitor=monitor,
        save_last=save_last,
        save_top_k=save_top_k,
        mode=mode,
        every_n_train_steps=every_n_train_steps,
        # verbose=True,
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = "{epoch}_last"
    return ckpt_callback

"""
This main entrance of fine-tuning for classification donwstream tasks
"""
import os
import random
# import torch
import pytorch_lightning as pl
# import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_log
from config import get_opt, args_list2dict
from dataset import DInterface
from models import build_model, BasicModel
from utils import (
    load_model_path_by_args, load_pretrain_path_by_args, load_ckpt,
    checkpoint_standardize,
    get_checkpoint_callback, MyLogger
)


def main(args, stream):
    if not args.class_wise_sampling and args.data_seed is None:
        args.data_seed = random.randint(1, 1e8)

    # * Prepare data_module *
    dm = DInterface(**vars(args))
    args.class_dict = dm.init_data['class_dict']
    args.classes = list(args.class_dict.keys())
    args.num_classes = len(args.class_dict)
    global_bs = args.gpus * args.batch_size if args.gpus > 1 else args.batch_size

    # * Build model *
    net = build_model(**vars(args))

    if args.load_pretrained:
        pretrained_path = load_pretrain_path_by_args(args, '.pth.tar')
        bl_layers = None
        if args.mode_name in ['train', 'finetune']:
            bl_layers = ['classifier', 'fc']
        net = load_ckpt(net, pretrained_path,
                        train=(args.mode_name == 'train'),
                        block_layers=bl_layers,
                        map_keys=args.map_keys,
                        verbose=True)

    model = BasicModel(net, **vars(args))

    # Resume
    load_path = load_model_path_by_args(args)
    if load_path is not None:
        model.load_from_checkpoint(checkpoint_path=load_path, strict=False)

    # * validate mode *
    if args.mode_name in ['val', 'test']:
        model.final_val = True
        trainer = Trainer.from_argparse_args(args)
        trainer.validate(model, datamodule=dm)
        return

    # * Callbacks *
    # Checkpoint callbacks
    if args.ckpt == 'debug' or not args.save_ckpt:
        # ckpt_callback = get_checkpoint_callback(args.ckpt, save_last=False, save_top_k=0)
        ckpt_callback = get_checkpoint_callback(f'Task_models/{args.net_suffix}', save_last=False, save_top_k=1)
    else:
        cpDir = '{}/{}_{}'.format(args.ckpt, args.model_name, args.net_suffix)
        every_n_train_steps = dm.num_samples//global_bs
        if args.ckpt_ever_n_epoch:
            every_n_train_steps *= args.ckpt_ever_n_epoch
        ckpt_callback = get_checkpoint_callback(
            cpDir, 'val/acc', 'max',
            filename='{epoch}_{val_acc:.2f}',
            every_n_train_steps=every_n_train_steps)

    # Logging callbacks
    if args.train_scale >= 1:
        version_str = f'{args.dataset}_ts={int(args.train_scale):d}'
    else:
        version_str = f'{args.dataset}_ts={args.train_scale:.2%}'
    logger_tb = pl_log.TensorBoardLogger(args.log_dir, args.exp_name, version_str)
    log_dir = logger_tb.log_dir
    args.logger = [logger_tb]
    if pl.utilities.distributed._get_rank() == 0:
        os.makedirs(log_dir)
        stream.all_to_file(log_dir+'/{}.log'.format(
            args.exp_name), flush=True)

    # logger_eren = MyLogger(log_dir, 'exp_log')
    logger_eren = MyLogger(None)
    args.progress_bar_refresh_rate = 0  # Use MyLogger() install of progress_bar
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    args.callbacks = [
        ckpt_callback, logger_eren, lr_monitor
    ]

    # * Accelerating *
    if args.gpus > 1 and (args.accelerator is None and args.plugins is None):
        args.accelerator = 'ddp'
    if args.accelerator == 'ddp':
        args.plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)

    if args.mode_name in ['train', 'finetune']:
        args.benchmark = True

    # * Begin training and testing *
    trainer = Trainer.from_argparse_args(args)

    # Begin training
    trainer.fit(model, datamodule=dm)

    # Final test
    model.final_val = True
    trainer.validate(model, ckpt_path='best', datamodule=dm)

    # Other operations
    print('Best ckpt: {}'.format(trainer.checkpoint_callback.best_model_path))
    if args.ckpt != 'debug' and args.save_ckpt:
        checkpoint_standardize(cpDir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--mode', default=4, type=int, help="Current main mode")
    parser.add_argument('--pj', default='', type=str)
    parser.add_argument('--exp_note', default='', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)

    # LR Scheduler
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--lr_scheduler', default='cosine',
                        choices=['step', 'cosine', 'warmup-anneal'], type=str)
    parser.add_argument('--lr_decay_steps', default=[60, 80], type=int, nargs='+')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--final_lr', default=0., type=float)
    parser.add_argument('--freeze_layers', type=str, nargs='+',
                        default=['features', 'ExcludeFC'])

    # Resume
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_expnum', default='', type=str)
    parser.add_argument('--load_epoch', default=None, type=int)
    parser.add_argument('--load_pretrained', type=str,
                        default='../TOV_models/0102300000_22014162253_pretrain')
    parser.add_argument('--map_keys', type=str, nargs='+',
                        # default=[])
                        default=['', 'features.'])
    parser.add_argument('--ckpt', type=str,
                        default='Task_models')
    parser.add_argument('--ckpt_ever_n_epoch', default=0, type=int)
    parser.add_argument('--save_ckpt', default=False, action='store_true')

    # Data Info
    parser.add_argument('--dataset', default='aid', type=str)
    parser.add_argument('--data_seed', default=0, type=int)

    parser.add_argument('--train_scale', default=20, type=float,
                        help="Scale of training set reduction, e.g. 0.1")
    parser.add_argument('--val_scale', default=None, type=float,
                        help="Scale of valing set reduction, e.g. 0.1")
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--class_wise_sampling', default=True, type=bool)
    parser.add_argument('--mean', type=float, nargs='+', default=None)
    parser.add_argument('--std', type=float, nargs='+', default=None)

    # Training Info
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--check_val_every_n_epoch', default=2, type=int)
    parser.add_argument('--log_dir', default='./Log', type=str)

    # Model Hyperparameters
    parser.add_argument('--model_name', default='010'+'2300'+'000', type=str)
    parser.add_argument('--pretrained', default=False, action='store_true')

    # Advanced Optimized Training
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--accelerator', default=None, type=str)
    parser.add_argument('--plugins', default=None, type=str)

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--super_prefetch', default=True, action='store_true')
    args = parser.parse_args()

    # Extra Arguments
    # args.num_sanity_val_steps = 1
    # Combine user configs with predefined configs
    args, stream = get_opt(args.dataset, args, True)
    print(args.exp_note)
    args.map_keys = args_list2dict(args.map_keys)
    # print(args.map_keys)

    main(args, stream)
    stream.close()

from collections import OrderedDict
import os
import torch
from tqdm import tqdm


# **************************************************
# **************** Checkpoint tools ****************
# **************************************************
def load_pretrain_path_by_args(args, ext='.pth.tar'):
    def generate_root():
        if os.path.isdir(args.load_pretrained):
            return args.load_pretrained
        else:
            return os.path.join(args.ckpt, args.load_pretrained)

    root = generate_root()
    if os.path.isfile(root):
        return root
    else:
        cp_files = os.listdir(root)
        cp_files = [name for name in cp_files if name.endswith(ext)]
        if args.load_epoch is None:
            if len(cp_files) == 1:
                name = cp_files[0]
            else:
                name = [name for name in cp_files if 'last' in name][0]
        else:
            name = 'epoch={}{}'.format(args.load_epoch, ext)
        return os.path.join(root, name)


def load_model_path_by_args(args):
    """ Resume model (.ckpt) for countiue learning. """
    def generate_root():
        if args.load_dir:
            return args.load_dir
        elif args.load_expnum:
            return os.path.join(args.ckpt, args.load_expnum)
        else:
            return None

    root = generate_root()
    if root is None:
        return None
    elif root.startswith('https'):
        return root
    elif not os.path.exists(root):
        return None

    if os.path.isfile(root):
        return root
    else:
        cp_files = os.listdir(root)
        cp_files = [name for name in cp_files if name.endswith('.ckpt')]
        if args.load_epoch is None:
            if len(cp_files) == 1:
                name = cp_files[0]
            else:
                name = [name for name in cp_files if 'last' in name][0]
        else:
            name = 'epoch={}.ckpt'.format(args.load_epoch)
        return os.path.join(root, name)


def checkpoint_dict_mapping(state_dict, map_keys={}):
    if map_keys:
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            for org_k, new_k in map_keys.items():
                if org_k:
                    if org_k in k:
                        k = k.replace(org_k, new_k)
                else:
                    k = new_k + k
            new_dict[k] = v
        return new_dict
    else:
        return state_dict


def load_ckpt(model, ckpt, train=False, block_layers=[], map_keys={}, map_loc='cpu', verbose=False):
    import copy

    print(f"Trying to load: {ckpt}\n")
    log_str = ''
    if os.path.isfile(ckpt) or str(ckpt).startswith("http"):
        block_layers = [block_layers] if type(block_layers) is str else block_layers
        model_dict = model.state_dict()
        model_dict_bk = copy.deepcopy(model_dict)

        # Load ckpt from local_path or cloud_url
        if os.path.isfile(ckpt):
            pretrained_dict = torch.load(ckpt, map_location=map_loc)
        elif str(ckpt).startswith("http"):
            pretrained_dict = torch.hub.load_state_dict_from_url(str(ckpt), map_location=map_loc)

        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

        log_str += 'Load ckpt info:\n'
        log_str += '\tModel parameters count: {}\n'.format(len(list(model_dict.keys())))
        log_str += '\tCkpt parameters count: {}\n'.format(len(list(pretrained_dict.keys())))

        if map_keys:
            for k1, k2 in map_keys.items():
                log_str += '\tMap {} to {}\n'.format(k1, k2)
        pretrained_dict = checkpoint_dict_mapping(pretrained_dict, map_keys)

        if block_layers and len(block_layers) > 0:
            filtered_dict = {}
            bl_count = 0
            for k, v in pretrained_dict.items():
                if any(bl_l in k for bl_l in block_layers):
                    bl_count += 1
                    continue
                filtered_dict[k] = v
            model_dict.update(filtered_dict)
            log_str += f'\tBlock layer names: {block_layers}\n'
            # log_str += '\tBlock layer names: {}\n'.format(block_layers)
            log_str += f'\tBlock parameters count: {bl_count}.\n'
        else:
            model_dict = pretrained_dict
        feedback = model.load_state_dict(model_dict, strict=False)
        log_str += '\tUnexpected keys:\n'
        log_str += '\t\t{}\n'.format(feedback.unexpected_keys)
        log_str += '\tMissing keys:\n'
        log_str += '\t\t{}\n'.format(feedback.missing_keys)

        # Count update params
        model_dict = model.state_dict()
        update_param_num = 0
        for k, v in model_dict.items():
            if not (model_dict_bk[k] == model_dict[k]).all():
                update_param_num += 1
        log_str += f'\tNum of update params: {update_param_num}\n'
        if verbose:
            print(log_str)
    else:
        if train:
            print('\tCheckpoint file (%s) is not exist, re-initializtion' % ckpt)
        else:
            raise ValueError('Failed to load model checkpoint (%s).' % ckpt)
    return model


# **************************************************
# **************** Other functions *****************
# **************************************************
def checkpoints_conversion_pl2pt(in_dir, out_dir=None, map_keys={}):
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cp_names = os.listdir(in_dir)
    for name in cp_names:
        if not name.endswith('.ckpt'):
            continue
        src_pth = os.path.join(in_dir, name)
        checkpoint = torch.load(src_pth)
        model_dict = checkpoint['state_dict']

        model_dict = checkpoint_dict_mapping(model_dict, map_keys)
        # sname = name.replace('ckpt', 'pth')

        model_dict = {'state_dict': model_dict}
        sname = name.replace('ckpt', 'pth.tar')

        dst_pth = os.path.join(out_dir, sname)
        torch.save(model_dict, dst_pth, _use_new_zipfile_serialization=False)
        print('Convert {} to {} in {}'.format(name, sname, in_dir))


def checkpoints_conversion(in_dir, out_dir=None, map_keys={}):
    print('map_keys:\n\t', map_keys)
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cp_names = os.listdir(in_dir)
    for name in cp_names:
        if not name.endswith('.pth.tar') and not name.endswith('.pth'):
            continue
        src_pth = os.path.join(in_dir, name)
        model_dict = torch.load(src_pth)
        if 'state_dict' in model_dict.keys():
            model_dict = model_dict['state_dict']

        model_dict = checkpoint_dict_mapping(model_dict, map_keys)
        # sname = name.replace('ckpt', 'pth')

        model_dict = {'state_dict': model_dict}
        # sname = name.replace('.pth.tar', '.ckpt')
        # sname = name.replace('.pth', '.ckpt')
        sname = name.replace('.pth', '.pth.tar')

        dst_pth = os.path.join(out_dir, sname)
        torch.save(model_dict, dst_pth, _use_new_zipfile_serialization=False)
        print('Convert {} to {} in {}'.format(name, sname, in_dir))


def checkpoint_standardize(cp_root, map_keys=None, update=False):
    """Standardizing the checkpoint by converting `.ckpt` & `pth` into `.pth.tar`,
    and tidying up the keys.
    """
    for cur_root, dirs, files in os.walk(cp_root):
        if len(files) == 0 or 'Log' in cur_root:
            continue

        print(f'Convert ckeckpoints in {cur_root}')
        for name in files:
            if not name.endswith('.ckpt'):  # and not name.endswith('.pth'):
                continue
            src_pth = os.path.join(cur_root, name)
            model_dict = torch.load(src_pth)
            if 'state_dict' in model_dict.keys():
                model_dict = model_dict['state_dict']
            model_dict_keys = list(model_dict.keys())

            # Modified the checkpoint
            if name.endswith('.ckpt'):
                # pretreatment
                if 'module.' in model_dict_keys[0]:
                    map_keys = {'module.': ''}
                    model_dict = checkpoint_dict_mapping(model_dict, map_keys)

                # pretrain in pytorch_lightning
                if 'online_network.encoder.' in model_dict_keys[0]:
                    map_keys = {'online_network.encoder.': ''}  # BYOL
                    model_dict = {k: v for k, v in model_dict.items() if 'target_network.' not in k}
                elif 'encoder.' in model_dict_keys[0]:
                    map_keys = {'encoder.': ''}  # SimCLR / SwAV
                elif 'encoder_q.features.' in model_dict_keys[0] or 'queue' in model_dict_keys[0]:
                    map_keys = {'encoder_q.features.': ''}  # MoCo
                    model_dict = {k: v for k, v in model_dict.items() if 'encoder_k.' not in k}
                elif 'encoder_q.' in model_dict_keys[0] and 'encoder_q.features.' not in model_dict_keys[0]:
                    map_keys = {'encoder_q.': ''}  # MoCo
                    model_dict = {k: v for k, v in model_dict.items() if 'encoder_k.' not in k}
                # train in pytorch_lightning
                elif 'model.features.' in model_dict_keys[0]:
                    map_keys = {'model.features.': ''}
                else:
                    print('Warning: Fail to match the fitable `map_keys`.')
                    continue
                sname = name.replace('.ckpt', '.pth.tar')
            elif name.endswith('.pth'):
                # pretrain in pytorch_lightning
                if 'features.' in model_dict_keys[0]:
                    map_keys = {'features.': ''}  # MoCo
                sname = name.replace('.pth', '.pth.tar')
            # else:
            #     continue
            dst_pth = os.path.join(cur_root, sname)
            if os.path.exists(dst_pth) and not update:
                continue

            model_dict = checkpoint_dict_mapping(model_dict, map_keys)
            model_dict = {'state_dict': model_dict}
            torch.save(model_dict, dst_pth, _use_new_zipfile_serialization=False)
            print('\t{} to {}'.format(name, sname))


def check_dataset():
    import cv2
    from tqdm import tqdm

    data_dir = '/data2/Classification/WHU_RSD46'

    for root, dirs, files in tqdm(os.walk(data_dir)):
        for name in files:
            try:
                img = cv2.imread(root+'/'+name)
                img.shape
                img = img[:, :, ::-1]  # BGR→RGB
                del img

            except AttributeError:
                print('{}/{} is broken!!!'.format(root, name))

if __name__ == "__main__":
    print('...')
    input('Main func: Please press the Enter key to proceed.')

    pass

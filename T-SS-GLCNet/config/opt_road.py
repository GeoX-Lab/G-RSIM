# coding:utf-8
'''
Default Config about cvpr_road数据集.
'''


class DefaultConfig(object):
    env = 'result'  # visdom 环境
    root = None
    dataset_dir = None
    ckpt = None
    outputimage = None
    benchmark = True
    deterministic = False
    enabled = True
    arch='resnet50'
    non_blocking=True
    pin_memory=True
    quick_train=True
    n_channels=None

    nodes=1
    ngpus_per_node=1
    world_size=1
    gpu=None
    #local_rank=-1

    pross_num = 16#pross_num_for_GLCNet


    continue_training = False

    # Optimiztion related arguments
    use_gpu = True  # if use GPU
    ckpt_freq = 40
    save_log=True

    lr_decay = 0.98  # pre epoch
    weight_decay = 1e-5  # L2 loss
    loss = ['MSELoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss', 'FocalLoss', 'SegmentationMultiLosses']
    
    dtype = 'RGB'  # default
    bl_dtype = ['']
    # Data related arguments
    num_workers = 8  # number of data loading workers#
    scnn_size = [32, 32]  # 56-448
    input_size = (256, 256)  # final input size of network(random-crop use this)
    crop_params = [256, 256, 128]  # [H, W, stride] only used by slide-crop
    crop_mode = 'slide'  # crop way of Val data, one of [random, slide]
    ont_hot = False  # Is the output of data_loader one_hot type
    class_num=None
    print_freq = 20  # print info every N batch


    small_sample = False  #
    class_dict_road = None
    environ = '0'  # select gpu
    device_id = [0]  # select gpu
    batch_size = 16 # batch size#
    start_epoch = 0  # use to continue_training
    cur_epoch = 0


opt = DefaultConfig()

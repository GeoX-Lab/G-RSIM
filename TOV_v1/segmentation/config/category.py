# -*- coding:utf-8 -*-


class CVPR_LandCover(object):
    ''' Category details of CVPR_LandCover dataset.

    | 场景/地物名称 | 标定标签 |
    | ----- | ----- |
    | Ignore           | 0|
    | Urban land       | 1|
    | Agriculture land | 2|
    | Rangeland        | 3|
    | Forest land      | 4|
    | Water            | 5|
    | Barren land      | 6|
    | Unknown          | 7|

    '''
    plan_name = 'CVPR_LandCover'
    table = {
        'Urban land': 0,
        'Agriculture land': 1,
        'Rangeland': 2,
        'Forest land': 3,
        'Water': 4,
        'Barren land': 5,
        'Unknown': 6,
    }

    names = [name for (name, _) in table.items()]
    inds = table.values()

    num = len(table)

    color_table = [
        # [5, 5, 5],  # 0
        [0, 255, 255],  # 0
        [255, 255, 0],  # 1
        [255, 0, 255],  # 2
        [0, 255, 0],  # 3
        [0, 0, 255],  # 4
        [255, 255, 255],  # 5
        [0, 0, 0],  # 6
    ]  # RGB

    bgr_table = [tb[::-1] for tb in color_table]  # BGR
    mapping = None


class ISPRS(object):
    ''' Category details of ISPRS dataset.

    |场景/地物名称|标定标签|
    |-----|-----|
    |BG (ignore)         | 0|
    |Impervious surfaces | 0|
    |Building            | 1|
    |Low vegetation      | 2|
    |Tree                | 3|
    |Car                 | 4|
    |Clutter/background  | 5|
    '''
    plan_name = 'ISPRS'
    names = [
        'Ignore',
        'Impervious surfaces',
        'Building',
        'Low vegetation',
        'Tree',
        'Car',
        'Clutter/background',
    ]
    table = {cls: i for (i, cls) in enumerate(names)}
    inds = list(range(len(names)))

    num = len(table)

    color_table = [
        [0, 0, 0],  # 0
        [255, 255, 255],  # 1
        [0, 0, 255],  # 2
        [0, 255, 255],  # 3
        [0, 255, 0],  # 4
        [255, 255, 0],  # 5
        [255, 0, 0],  # 6
    ]  # RGB

    bgr_table = [tb[::-1] for tb in color_table]  # BGR
    mapping = None


class DLRSD(object):
    ''' Category details of DLRSD dataset. '''
    table = {
        "none": 0,
        "airplane": 1,
        "bare soil": 2,
        "buildings": 3,
        "cars": 4,
        "chaparral": 5,
        "court": 6,
        "dock": 7,
        "field": 8,
        "grass": 9,
        "mobile home": 10,
        "pavement": 11,
        "sand": 12,
        "sea": 13,
        "ship": 14,
        "tanks": 15,
        "trees": 16,
        "water": 17,
    }
    names = [name for (name, _) in table.items()]
    inds = table.values()

    num = len(table)

    color_table = [
        [0, 0, 0],
        [166, 202, 240],
        [128, 128, 0],
        [0, 0, 128],
        [255, 0, 0],
        [0, 128, 0],
        [128, 0, 0],
        [255, 233, 233],
        [160, 160, 164],
        [0, 128, 128],
        [90, 87, 255],
        [255, 255, 0],
        [255, 192, 0],
        [0, 0, 255],
        [255, 0, 192],
        [128, 0, 128],
        [0, 255, 0],
        [0, 255, 255],
    ]  # RGB

    bgr_table = [tb[::-1] for tb in color_table]  # BGR
    mapping = None

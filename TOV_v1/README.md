# Project of TOV 1.0
This is project of the TOV paper:
```
@Article{chen2020mocov2,
  author  = {Chao Tao, Ji Qi, Guo Zhang, Qing Zhu, Weipeng Lu and Haifeng Li},
  title   = {TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-supervised Learning},
  journal = {},
  year    = {2022},
}
```

## Introduction
**TOV**: **T**he **O**rginal **V**ision for Optical Remote Sensing Image Understanding

> We argue that a more desirable remote sensing image understanding (RSIU) model should be trained with intrinsic structure from data rather than extrinsic human labels to realize generalizability across a wide range of RSIU tasks. According to this hypothesis, we define the original vision model, which serves as a general purpose of visual perception for a wide range of RSIU tasks, and proposed a framework to build the first original vision model (TOV 1.0) in remote sensing filed.

To foster the development of an original vision model for RSIU, in this project, we will realse our pre-trained TOV model and related materials:
- [x] Pretrained TOV model ([GoogleDrive](https://drive.google.com/drive/folders/14c0TnHFi1N_DC_egcoNWHCKX9C2pmmUR?usp=sharing) | [BaiduDrive](https://pan.baidu.com/s/1NHnuTbj7fVvCuUJXU9N5vQ?pwd=TOV1))
- [x] The benchmark datasets and codes for evalutation.
- [ ] TOV-RS-balanced: an remote sensing image dataset constructed by the proposed data sampling. ([BaiduDrive (Soon to be released...)]())
- [ ] ...

## Using TOV model for various downstream RSIU tasks
TOV pre-trained models are expected to be placed in the `TOV_models` folder, e.g., `TOV_models/0102300000_22014162253_pretrain/TOV_v1_model_pretrained_on_TOV-RS-balanced_ep800.pth.tar`

### Classification
#### Example: fine-tune pre-trained TOV model on AID
```bash
python classification/main_cls.py \
    --dataset aid \
    --train_scale 5 \  # use 5 samples per category for finetune

# Other parameters (e.g., `learning_rate` can directly use the default values provided in the `classification/main_cls.py`
```
The scene classification datasets are expected to be placed in the `classification/Cls_data` directory, e.g., `classification/Cls_data/AID`

### Object detection
We recommend using the powerful object detection framework, [MMDetection](https://github.com/open-mmlab/mmdetection).

### Semantic segmentation
#### Example: fine-tune pre-trained TOV model on DLRSD
```bash
pythonsegmentation/main_seg.py \
    --dataset dlrsd \
    --train_scale 0.01 \  # use 1% training samples for finetune
    --batch_size 16 --gpus 2
```
The semantic segmentation datasets are expected to be placed in the 
`segmentation/Seg_data` directory, e.g., `segmentation/Seg_data/ISPRS_Postdam`

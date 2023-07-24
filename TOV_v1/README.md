# Project of TOV 1.0
This is project of the TOV paper:
```
[1] C. Tao, J. Qi, G. Zhang, Q. Zhu, W. Lu and H. Li, "TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-Supervised Learning," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 4916-4930, 2023, doi: 10.1109/JSTARS.2023.3271312.

@ARTICLE{10110958,
  author={Tao, Chao and Qi, Ji and Zhang, Guo and Zhu, Qing and Lu, Weipeng and Li, Haifeng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-Supervised Learning}, 
  year={2023},
  volume={16},
  number={},
  pages={4916-4930},
  doi={10.1109/JSTARS.2023.3271312}
}
```
## Abstract
Do we on the right way for remote sensing image understanding (RSIU) by training models via supervised data-dependent and task-dependent way, instead of human vision in a label-free and task-independent way? We argue that a more desirable RSIU model should be trained with intrinsic structure from data rather that extrinsic human labels to realize generalizability across a wide range of RSIU tasks. According to this hypothesis, we proposed \textbf{T}he \textbf{O}riginal \textbf{V}ision model (TOV) in remote sensing filed. Trained by massive unlabeled optical data along a human-like self-supervised learning (SSL) path that is from general knowledge to specialized knowledge, TOV model can be easily adapted to various RSIU tasks, including scene classification, object detection, and semantic segmentation, and outperforms dominant ImageNet supervised pretrained method as well as two recently proposed SSL pretrained methods on majority of 12 publicly available benchmarks. Moreover, we analyze the influences of two key factors on the performance of building TOV model for RSIU, including the influence of using different data sampling methods and the selection of learning paths during self-supervised optimization. We believe that a general model which is trained by a label-free and task-independent way may be the next paradigm for RSIU and hope the insights distilled from this study can help to foster the development of an original vision model for RSIU.


## Introduction
**TOV**: **T**he **O**rginal **V**ision for Optical Remote Sensing Image Understanding

> We argue that a more desirable remote sensing image understanding (RSIU) model should be trained with intrinsic structure from data rather than extrinsic human labels to realize generalizability across a wide range of RSIU tasks. According to this hypothesis, we define the original vision model, which serves as a general purpose of visual perception for a wide range of RSIU tasks, and proposed a framework to build the first original vision model (TOV 1.0) in remote sensing filed.

To foster the development of an original vision model for RSIU, in this project, we will realse our pre-trained TOV model and related materials:
- [x] Pretrained TOV model ([GoogleDrive](https://drive.google.com/drive/folders/14c0TnHFi1N_DC_egcoNWHCKX9C2pmmUR?usp=sharing) | [BaiduDrive](https://pan.baidu.com/s/1NHnuTbj7fVvCuUJXU9N5vQ?pwd=TOV1))
- [x] The benchmark datasets and codes for evalutation.
- [ ] TOV-RS: the large scale remote sensing image dataset constructed by the proposed method. ([BaiduDrive](https://pan.baidu.com/s/1VGvoi8UlgbBrFkWmWsORvQ?pwd=xy29))
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

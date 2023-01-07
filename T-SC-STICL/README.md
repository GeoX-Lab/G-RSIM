## STICL: Spatial-temporal Invariant Contrastive Learning for Remote Sensing Scene Classification

This is a PyTorch implementation of the [STICL](https://ieeexplore.ieee.org/document/9770815):
```
@ARTICLE{huang2022sticl,
  author={Huang, Haozhe and Mou, Zhongfeng and Li, Yunying and Li, Qiujun and Chen, Jie and Li, Haifeng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Spatial-Temporal Invariant Contrastive Learning for Remote Sensing Scene Classification}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3173419}}
```

### Details
This version of sticl is implemented based on [Moco v2](https://github.com/facebookresearch/moco), and spatial-temporal Invariant Contrastive Learning can also be used in other self-supervised learning methods. The framework of STICL consists of two main parts, the pre-training phase and the fine-tuning phase.

#### **Pretraining**

For example, to do pre-training of a ResNet-50 model on unlabeled RSIs, run:
```
python pretrain.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --epochs 300 \
  --mlp \
  --moco-t 0.2 \
  --save_model 'mocov2_mix_bs256_300e_sti_10p.pth.tar' \
  --cos \
  --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0 \
  --st_prob 0.1 \
  --data [your unlabeled image folder with train and val folders]
  
```
--st_prob (between 0 and 1) for adjusting the strength of Spatial-temporal feature transfer.

#### **Finetuning**

With a pre-trained model, to do finetuning on labeled RSIs, run:
```
python finetune.py \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 256 \
  --method 'moco' \
  --pretrained [your pre-trained model]\
  --num_samples 20 \
  --nclass 30 \
  --data  [your labeled image folder with train and val folders]
```
--num_samples is used to set how many samples per class are used for finetuning.

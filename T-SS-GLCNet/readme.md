# Remote Sensing Images Semantic Segmentation with General Remote Sensing Vision Model via a Self-Supervised Contrastive Learning Method
## Abstract


Recently, supervised deep learning has achieved great success in remote sensing image (RSI) semantic segmentation. However, supervised learning for semantic segmentation requires a large number of labeled samples, which is difficult to obtain in the field of remote sensing. A new learning paradigm, self-supervised learning (SSL), can be used to solve such problems by pre-training a general model with a large number of unlabeled images and then fine-tuning it on a downstream task with very few labeled samples. Contrastive learning is a typical method of SSL that can learn general invariant features. However, most existing contrastive learning methods are designed for classification tasks to obtain an image-level representation, which may be suboptimal for semantic segmentation tasks requiring pixel-level discrimination. Therefore, we propose a global style and local matching contrastive learning network (GLCNet) for remote sensing image semantic segmentation. Specifically, 1) the global style contrastive learning module is used to better learn an image-level representation, as we consider that style features can better represent the overall image features. 2) The local features matching contrastive learning module is designed to learn representations of local regions, which is beneficial for semantic segmentation. We evaluate four RSI semantic segmentation datasets, and the experimental results show that our method mostly outperforms state-of-the-art self-supervised methods and the ImageNet pre-training method. Specifically, with 1\% annotation from the original dataset, our approach improves Kappa by 6\% on the ISPRS Potsdam dataset relative to the existing baseline. Moreover, our method outperforms supervised learning methods when there are some differences between the datasets of upstream tasks and downstream tasks. Our study promotes the development of self-supervised learning in the field of RSI semantic segmentation. Since SSL could directly learn the essential characteristics of data from unlabeled data, which is easy to obtain in the remote sensing field, this may be of great significance for tasks such as global mapping. 

You can visit the paper via 

## Dataset Directory Structure
-------
File Structure is as follows:   

    $train_RGBIR/*.tif     
    $train_lbl/*.tif     
    $val_RGBIR/*.tif      
    $val_lbl/*.tif    
    train_RGBIR.txt    
    trainR1_RGBIR.txt     
    trainR1_lbl.txt       
    val_RGBIR.txt       
    val_lbl.txt
    
## Training
-------         
To pretrain the model with our GLCNet and finetune , try the following command:      
```
python main_ss.py  root=./data_example/Potsdam
    --ex_mode=1  --self_mode=1 \  
    --self_max_epoch=400  --ft_max_epoch=150 \
    --self_data_name=train  --ft_train_name=trainR1
```   
    
## Citation
If our repo is useful to you, please cite our published paper as follow:

```
Bibtex
@article{Li2021GLCNet,
    title={Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images},
    author={Li, Haifeng and Yi, Li and Zhang, Guo and Liu, Ruoyun and Huang, Haozhe and Zhu, Qing and Tao, Chao},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    DOI = {10.1109/TGRS.2022.3147513},
    year={2022},
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Li, Haifeng
%A Yi, Li
%A Zhang, Guo
%A Liu, Ruoyun
%A Huang, Haozhe
%A Zhu, Qing
%A Tao, Chao
%D 2022
%T Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images
%B IEEE Transactions on Intelligent Transportation Systems
%R DOI:10.1109/TITS.2019.2935152
%! Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images
```

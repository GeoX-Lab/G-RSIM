# Remote Sensing Images Semantic Segmentation with General Remote Sensing Vision Model via a Self-Supervised Contrastive Learning Method
## Abstract


Recently, supervised deep learning has achieved great success in remote sensing images (RSIs) semantic segmentation. However, supervised learning for semantic segmentation requires a large number of labeled samples which is difficult to obtain in the field of remote sensing. A new learning paradigm, self-supervised learning (SSL), can be used to solve such problems by pre-training a general model with large unlabeled images and then fine-tuning on a downstream task with very few labeled samples. Contrastive learning is a typical method of SSL, which can learn general invariant features. However, most of the existing contrastive learning is designed for classification tasks to obtain image-level representation, which may be sub-optimal for semantic segmentation tasks requiring pixel-level discrimination. Therefore, we propose Global style and Local matching Contrastive Learning Network (GLCNet) for remote sensing semantic segmentation. Specifically, 1) The global style contrastive module is used to learn image-level representation better, as we consider the style features can better represent the overall image features. 2) local features matching contrastive module is designed to learn representation of local regions which is beneficial for semantic segmentation. We evaluate on four remote sensing semantic segmentation datasets, and the experimental results show that our method mostly outperforms state-of-the-art self-supervised methods and ImageNet pre-training. Specifically, with 1% annotation from the original dataset, our approach improves Kappa by 6% on the ISPRS Potsdam dataset and 3% on Deep Globe Land Cover Classification dataset relative to the existing baseline. Moreover, our method outperforms supervised learning when there are some differences between the data sets of upstream tasks and downstream tasks. Our study promotes the development of self-supervised learning in the field of remote sensing semantic segmentation. Since SSL could directly learn the essential characteristics of data from unlabeled data which is easy to obtain in remote sensing filed, this may be of great significance for tasks such as global mapping

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
    title={Remote Sensing Images Semantic Segmentation with General Remote Sensing Vision Model via a Self-Supervised Contrastive Learning Method},
    author={Li, Haifeng and Yi, Li and Zhang, Guo and Liu, Ruoyun and Huang, Haozhe and Zhu, Qing and Tao, Chao},
    journal={arxiv.org},
    DOI = {arxiv.org},
    year={2021},
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
%D 2021
%T Remote Sensing Images Semantic Segmentation with General Remote Sensing Vision Model via a Self-Supervised Contrastive Learning Method
%B arxiv.org
%R arxiv.org
%! Remote Sensing Images Semantic Segmentation with General Remote Sensing Vision Model via a Self-Supervised Contrastive Learning Method
```
       


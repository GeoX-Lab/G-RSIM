# G-RSIM

## Overview
This repository includes our works on General remote sensing image understanding.

**The full list of works is as follows:**
- **Literature review article** titled "Remote senslng image intelligent interpretation: from supervised learning to self-supervised learning".
   - Links: [[Paper](https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C44YLTlOAiTRKibYlV5Vjs7iy_Rpms2pqwbFRRUtoUImHSE7i7PWr6_rbXYCr-h0wKFAkAPv5rjqr2HwmJN6B84s&uniplatform=NZKPT)]
   - Publication: 2021 Acta Geodaetica et Cartographica Sinica (测绘学报)
- **Literature review article** titled "Self-supervised remote sensing feature learning: Learning Paradigms, Challenges, and Future Works".
   - Links: [[Paper](https://ieeexplore.ieee.org/document/10126079)]
   - Publication: 2023 IEEE Transactions on Geoscience and Remote Sensing

- A pioneering investigation of self-supervised learning paradigms in the remote sensing filed.
   - Links: [[Paper](https://ieeexplore.ieee.org/document/9284640/) | [Code](https://github.com/ErenTuring/RSISC_SSL_Paradigm)]
   - Citation: Tao, C., Qi, J., Lu, W., Wang, H., & Li, H. (2020). Remote Sensing Image Scene Classification With Self-Supervised Paradigm Under Limited Labeled Samples. IEEE Geoscience and Remote Sensing Letters, 19, 1–5. https://doi.org/10.1109/LGRS.2020.3038420

- **T-SS-GLCNet**: a global style and local matching contrastive learning network (GLCNet) for remote sensing image semantic segmentation.
   - Links: [[Paper](https://ieeexplore.ieee.org/document/9696319/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/T-SS-GLCNet)]

- **T-SC-STICL**: a spatial-temporal invariant contrastive learning framework to learn spatial-temporal invariant representations from unlabeled images containing a large number of spatial-temporal scenes.
   - Links: [[Paper](https://ieeexplore.ieee.org/document/9770815/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/T-SC-STICL)]
   - Publication: 2022 IEEE Geoscience and Remote Sensing Letters

- **TOV_v1**: the Original Vision Model for Optical Remote Sensing Image Understanding via Self-Supervised Learning.
   - Links: [[Paper](https://ieeexplore.ieee.org/document/10110958/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)]
   - Publication: 2023 IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing

- **FALSE**: False Negative Samples Aware Contrastive Learning for Semantic Segmentation of High-Resolution Remote Sensing Image
   - Links: [[Paper](https://ieeexplore.ieee.org/document/9954056/) | [Code](https://github.com/GeoX-Lab/FALSE)]
   - Publication: 2022 IEEE Geoscience and Remote Sensing Letters

- **LaST**: Label-Free Self-Distillation Contrastive Learning With Transformer Architecture for Remote Sensing Image Scene Classification
   - Links: [[Paper](https://ieeexplore.ieee.org/document/9802117/) | [Code]()]
   - Publication: 2022 IEEE Geoscience and Remote Sensing Letters


## Details of Literature Review Article

### 2021
**Title**

*Remote sensing image intelligent interpretation: from supervised learning to self-supervised learning*

**Citation**

- 陶超,阴紫薇,朱庆,等.遥感影像智能解译:从监督学习到自监督学习[J].测绘学报,2021,50(8):1122-1134.DOI:10.11947/j. AGCS.2021.20210089.
- TAO Chao,YINZiwei,ZHU Qing,etal.Remote sensing image intelligent interpretation: from supervised learning to self-supervised learning[J].Acta Geodaetica et Cartographica Sinica,2021,50(8):1122-1134.DOI:10.11947/j.AGCS.2021.20210089

### 2023
**Title**

*Self-supervised remote sensing feature learning: Learning Paradigms, Challenges, and Future Works*

**Citation**

Chao Tao, Ji Qi,  Mingning Guo, Qing Zhu, and Haifeng Li, Self-supervised remote sensing feature learning: Learning Paradigms, Challenges, and Future Works，IEEE Transactions on Geoscience and Remote Sensing，DOI: 10.1109/TGRS.2023.3276853

```bibteX
@ARTICLE{10126079,
  author={Tao, Chao and Qi, Ji and Guo, Mingning and Zhu, Qing and Li, Haifeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Self-supervised Remote Sensing Feature Learning: Learning Paradigms, Challenges, and Future Works}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3276853}}
```

## Details of Proposed Methods

### (2021) T-SS-GLCNet
**Abstract**

Recently, supervised deep learning has achieved a great success in remote sensing image (RSI) semantic segmentation. However, supervised learning for semantic segmentation requires a large number of labeled samples, which is difficult to obtain in the field of remote sensing. A new learning paradigm, self-supervised learning (SSL), can be used to solve such problems by pretraining a general model with a large number of unlabeled images and then fine-tuning it on a downstream task with very few labeled samples. Contrastive learning is a typical method of SSL that can learn general invariant features. However, most existing contrastive learning methods are designed for classification tasks to obtain an image-level representation, which may be suboptimal for semantic segmentation tasks requiring pixel-level discrimination. Therefore, we propose a global style and local matching contrastive learning network (GLCNet) for RSI semantic segmentation. Specifically, first, the global style contrastive learning module is used to better learn an image-level representation, as we consider that style features can better represent the overall image features. Next, the local features matching the contrastive learning module is designed to learn the representations of local regions, which is beneficial for semantic segmentation. We evaluate four RSI semantic segmentation datasets, and the experimental results show that our method mostly outperforms the state-of-the-art self-supervised methods and the ImageNet pretraining method. Specifically, with 1% annotation from the original dataset, our approach improves Kappa by 6% on the International Society for Photogrammetry and Remote Sensing (ISPRS) Potsdam dataset relative to the existing baseline. Moreover, our method outperforms supervised learning methods when there are some differences between the datasets of upstream tasks and downstream tasks. Our study promotes the development of SSL in the field of RSI semantic segmentation. Since SSL could directly learn the essential characteristics of data from unlabeled data, which is easy to obtain in the remote sensing field, this may be of great significance for tasks such as global mapping.

**Links**

[[Paper](https://ieeexplore.ieee.org/document/9696319/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/T-SS-GLCNet)]

**Citation**

Li, H., Li, Y., Zhang, G., Liu, R., Huang, H., Zhu, Q., & Tao, C. (2022). Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–14. https://doi.org/10.1109/tgrs.2022.3147513

```bibteX
@ARTICLE{9696319,
  author={Li, Haifeng and Li, Yi and Zhang, Guo and Liu, Ruoyun and Huang, Haozhe and Zhu, Qing and Tao, Chao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2022.3147513}}
```

### (2022) T-SC-STICL
**Abstract**

Self-supervised learning achieves close to supervised learning results on remote sensing image (RSI) scene classification. This is due to the current popular self-supervised learning methods that learn representations by applying different augmentations to images and completing the instance discrimination task which enables convolutional neural networks (CNNs) to learn representations invariant to augmentation. However, RSIs are spatial-temporal heterogeneous, which means that similar features may exhibit different characteristics in different spatial-temporal scenes. Therefore, the performance of CNNs that learn only representations invariant to augmentation still degrades for unseen spatial-temporal scenes due to the lack of spatial-temporal invariant representations. We propose a spatial-temporal invariant contrastive learning (STICL) framework to learn spatial-temporal invariant representations from unlabeled images containing a large number of spatial-temporal scenes. We use optimal transport to transfer an arbitrary unlabeled RSI into multiple other spatial-temporal scenes and then use STICL to make CNNs produce similar representations for the views of the same RSI in different spatial-temporal scenes. We analyze the performance of our proposed STICL on four commonly used RSI scene classification datasets, and the results show that our method achieves better performance on RSIs in unseen spatial-temporal scenes compared to popular self-supervised learning methods. Based on our findings, it can be inferred that spatial-temporal invariance is an indispensable property for a remote sensing model that can be applied to a wider range of remote sensing tasks, which also inspires the study of more general remote sensing models.

**Links**

[[Paper](https://ieeexplore.ieee.org/document/9770815/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/T-SC-STICL)]

**Citation**

Citation: Huang, H., Mou, Z., Li, Y., Li, Q., Chen, J., & Li, H. (2022). Spatial-Temporal Invariant Contrastive Learning for Remote Sensing Scene Classification. IEEE Geoscience and Remote Sensing Letters, 19, 1–5. https://doi.org/10.1109/LGRS.2022.3173419

```bibteX
@ARTICLE{9770815,
  author={Huang, Haozhe and Mou, Zhongfeng and Li, Yunying and Li, Qiujun and Chen, Jie and Li, Haifeng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Spatial-Temporal Invariant Contrastive Learning for Remote Sensing Scene Classification}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3173419}}
```

### (2022) TOV_v1
**Abstract**

Are we on the right way for remote sensing image understanding (RSIU) by training models in a supervised data-dependent and task-dependent manner, instead of original human vision in a label-free and task-independent way? We argue that a more desirable RSIU model should be trained with intrinsic structure from data rather than extrinsic human labels to realize generalizability across a wide range of RSIU tasks. According to this hypothesis, we proposed T he O riginal V ision model (TOV) in the remote sensing field. Trained by massive unlabeled optical data along a human-like self-supervised learning (SSL) path that is from general knowledge to specialized knowledge, TOV model can be easily adapted to various RSIU tasks, including scene classification, object detection, and semantic segmentation, and outperforms dominant ImageNet supervised pre-trained method as well as two recently proposed SSL pre-trained methods on the majority of 12 publicly available benchmarks. Moreover, we analyze the influences of two key factors on the performance of building TOV model for RSIU, including the influence of using different data sampling methods and the selection of learning paths during self-supervised optimization. We believe that a general model which is trained in a label-free and task-independent way may be the next paradigm for RSIU and hope the insights distilled from this study can help to foster the development of an original vision model for RSIU.

**Links**

[[Paper](https://ieeexplore.ieee.org/document/10110958/) | [Code](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)]

**Citation**

Citation: Tao, C., Qi, J., Zhang, G., Zhu, Q., Lu, W., & Li, H. (2023). TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-Supervised Learning. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 1–16. https://doi.org/10.1109/JSTARS.2023.3271312

```bibteX
@ARTICLE{10110958,
  author={Tao, Chao and Qi, Ji and Zhang, Guo and Zhu, Qing and Lu, Weipeng and Li, Haifeng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-Supervised Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/JSTARS.2023.3271312}}
```

### (2022) FALSE
**Abstract**

Self-supervised contrastive learning (SSCL) is a potential learning paradigm for learning remote sensing image (RSI)-invariant features through the label-free method. The existing SSCL of RSI is built based on constructing positive and negative sample pairs. However, due to the richness of RSI ground objects and the complexity of the RSI contextual semantics, the same RSI patches have the coexistence and imbalance of positive and negative samples, which causes the SSCL pushing negative samples far away while pushing positive samples far away, and vice versa. We call this the sample confounding issue (SCI). To solve this problem, we propose a False negAtive sampLes aware contraStive lEarning model (FALSE) for the semantic segmentation of high-resolution RSIs. Since SSCL pretraining is unsupervised, the lack of definable criteria for false negative sample (FNS) leads to theoretical undecidability, and we designed two steps to implement the FNS approximation determination: coarse determination of FNS and precise calibration of FNS. We achieve coarse determination of FNS by the FNS self-determination (FNSD) strategy and achieve calibration of FNS by the FNS confidence calibration (FNCC) loss function. Experimental results on three RSI semantic segmentation datasets demonstrated that the FALSE effectively improves the accuracy of the downstream RSI semantic segmentation task compared with the current three models, which represent three different types of SSCL models. The mean intersection over union (mIoU) on the ISPRS Potsdam dataset is improved by 0.7% on average; on the CVPR DGLC dataset, it is improved by 12.28% on average; and on the Xiangtan dataset, this is improved by 1.17% on average. This indicates that the SSCL model has the ability to self-differentiate FNS and that the FALSE effectively mitigates the SCI in SSCL.

**Links**

[[Paper](https://ieeexplore.ieee.org/document/9954056/) | [Code](https://github.com/GeoX-Lab/FALSE)]

**Citation**

Zhang, Z., Wang, X., Mei, X., Tao, C., & Li, H. (2022). FALSE: False Negative Samples Aware Contrastive Learning for Semantic Segmentation of High-Resolution Remote Sensing Image. IEEE Geoscience and Remote Sensing Letters, 19, 1–5. https://doi.org/10.1109/LGRS.2022.3222836

```bibteX
@ARTICLE{9954056,
  author={Zhang, Zhaoyang and Wang, Xuying and Mei, Xiaoming and Tao, Chao and Li, Haifeng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={FALSE: False Negative Samples Aware Contrastive Learning for Semantic Segmentation of High-Resolution Remote Sensing Image}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3222836}}
```

### (2022) LaST
**Abstract**

The increase in self-supervised learning (SSL), especially contrastive learning, has enabled one to train deep neural network models with unlabeled data for remote sensing image (RSI) scene classification. Nevertheless, it still suffers from the following issues: 1) the performance of the contrastive learning method is significantly impacted by the hard negative sample (HNS) issue, since the RSI scenario is complex in semantics and rich in surface features; 2) the multiscale characteristic of RSI is missed in the existing contrastive learning methods; and 3) as the backbone of a deep learning model, especially in the case of limited annotation, a convolutional neural network (CNN) does not include the adequate receptive field of convolutional kernels to capture the broad contextual information of RSI. In this regard, we propose label-free self-distillation contrastive learning with a transformer architecture (LaST). We introduce the self-distillation contrastive learning mechanism to address the HNS issue. Specifically, the LaST architecture comprises two modules: scale alignment with a multicrop module and a long-range dependence capture backbone module. In the former, we present global–local crop and scale alignment to encourage local-to-global correspondence and acquire multiscale relations. Then, the distorted views are fed into a transformer as a backbone, which is good at capturing the long-range-dependent contextual information of the RSI while maintaining the spatial smoothness of the learned features. Experiments on public datasets show that in the downstream scene classification task, LaST improves the performance of the self-supervised trained model by a maximum of 2.18% compared to the HNS-impacted contrastive learning approaches, and only 1.5% of labeled data can achieve the performance of supervised training CNNs with 10% labeled data. Moreover, this letter supports the integration of a transformer architecture and self-supervised paradigms in RSI interpretation.

**Links**

[[Paper](https://ieeexplore.ieee.org/document/9802117/) | [Code]()]

**Citation**

Wang, X., Zhu, J., Yan, Z., Zhang, Z., Zhang, Y., Chen, Y., & Li, H. (2022). LaST: Label-Free Self-Distillation Contrastive Learning With Transformer Architecture for Remote Sensing Image Scene Classification. IEEE Geoscience and Remote Sensing Letters, 19, 1–5. https://doi.org/10.1109/LGRS.2022.3185088

```bibteX
@ARTICLE{9802117,
  author={Wang, Xuying and Zhu, Jiawei and Yan, Zhengliang and Zhang, Zhaoyang and Zhang, Yunsheng and Chen, Yansheng and Li, Haifeng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={LaST: Label-Free Self-Distillation Contrastive Learning With Transformer Architecture for Remote Sensing Image Scene Classification}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3185088}}
```

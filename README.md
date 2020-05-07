# EHANet: An effective hierarchical aggregation network for face parsing

## Abstract
In recent years, benefiting from deep convolutional neural networks (DCNNs), face parsing has developed rapidly. However, it still has the following problems: (1) Existing state-of-the-art frameworks usually do not satisfy real-time while pursuing performance; (2) Similar appearances cause incorrect pixel label assignments, especially in the boundary; (3) To promote multi-scale prediction, deep features and shallow features are used for fusion without considering the semantic gap between them. To overcome these drawbacks, we propose an effective and efficient hierarchical aggregation network called EHANet for fast and accurate face parsing. More specifically, we first propose a Stage Contextual Attention Mechanism (SCAM), which uses higher-level contextual information to re-encoding the channel according to its importance. Secondly, a Semantic Gap Compensation Block (SGCB) is presented to ensure the effective aggregation of hierarchical information. Thirdly, the advantages of weighted boundary-aware loss effectively make up for the ambiguity of boundary semantics. Without any bells and whistles, combined with a lightweight backbone, we achieve outstanding results on both CelebAMask-HQ (78.19% mIoU) and Helen datasets (90.7% F1-score). Furthermore, our model can achieve 55 FPS on a single GTX 1080Ti card with 640 x 640 input and further reach over 300 FPS with a resolution of 256 x 256, which is suitable for real-world applications. 

## Visual Results
<div><div align=center>
  <img src="https://github.com/JACKYLUO1991/FaceParsing/blob/master/deployment/result/images/228.jpg" width="300" height="300" alt="raw"/>
<img src="https://github.com/JACKYLUO1991/FaceParsing/blob/master/deployment/result/renders/2.png" width="300" height="300" alt="pred"/></div>
  
## Thanks CelebAMask-HQ dataset
```
@article{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  journal={arXiv preprint arXiv:1907.11922},
  year={2019}
}
@article{luo2020ehanet,
  title={EHANet: An Effective Hierarchical Aggregation Network for Face Parsing},
  author={Luo, Ling and Xue, Dingyu and Feng, Xinglong},
  journal={Applied Sciences},
  volume={10},
  number={9},
  pages={3135},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

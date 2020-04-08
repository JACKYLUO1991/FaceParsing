# EHANet: An effective hierarchical aggregation network for face parsing

## abstract
Face parsing, as a branch of fine-grained segmentation, has received widespread attention due to its extensive application potentials. With the continuous progress of deep learning, a variety of model architectures have been derived. However, a simple element-wise concatenation (Concatenation) or element-wise sum (Add) operation is usually used between different feature layers to aggregate multi-scale context features, without considering the semantic gap between them. At the same time, similar appearances cause incorrect pixel label assignments, especially in the boundry. To overcome these drawbacks, we propose a Semantic Gap Compensation Block (SGCB) to ensure the effective aggregation of hierarchical information. In the meanwhile, we also propose a Stage Contextual Attention Mechanism (SCAM), which uses contextual information to reencoding the channel according to its importance. The advantages of weighted boundary-aware loss effectively make up for the ambiguity of boundary semantics. Benefiting from the superiority of our proposed components, without any bells and whistles, we achieve comparable results on both CelebAMask-HQ (78.19% Mean-IoU) and Helen datasets (90.7% F-Measure). In addition, our model can achieve over 300 FPS on a single GTX 1080Ti card with 256 x 256 input.}

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
```

# AAN: Appearance Adjustment Network for Medical Image Registration
Deformable image registration is fundamental for many medical image analyses. A key obstacle for accurate image registration is the variations in image appearance such as the variations in textures, intensities, and noises. These variations are highly apparent in medical images, especially in brain images where registration is frequently used. Recently, deep learning-based registration methods (DLRs), using deep neural networks, have gained computational efficiency that is several orders of magnitude faster than traditional optimization-based registration methods (ORs). DLRs rely on a globally optimized network that is trained with a set of training samples to achieve faster registration. However, DLRs tend to disregard the target-pair-specific optimization that is inherent in ORs. Consequently, DLRs have degraded adaptability to appearance variations and perform poorly when image pairs (fixed/moving images) have large appearance variations. In this study, we propose an Appearance Adjustment Network (AAN) to enhance DLRs’ adaptability to appearance variations. Our AAN, when integrated into a DLR, provides appearance transformations to reduce the appearance variations during registration. In addition, we propose an anatomy- constrained loss function through which our AAN can generate anatomy-preserving transformations. Our AAN has been purposely designed to be readily inserted into a wide range of DLRs and can be trained cooperatively in an unsupervised and end-to-end manner.

## Workflow
![workflow](https://github.com/MungoMeng/Image-Registration-AAN/blob/master/Figure/workflow.png)

## Publication
If this repository helps your work, please kindly cite our paper as follows:

* **Mingyuan Meng, Lei Bi, Michael Fulham, David Dagan Feng, Jinman Kim, "Enhancing Medical Image Registration via Appearance Adjustment Networks," arXiv:2103.05213 (under review). [[arXiv](https://arxiv.org/abs/2103.05213)]**

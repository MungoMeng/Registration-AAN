# AAN: Appearance Adjustment Network for Medical Image Registration
Image registration is fundamental for many medical image analyses. A key obstacle for accurate image registration is the variations in image appearance. Recently, deep learning-based registration methods (DLRs), using deep neural networks, have computational efficiency that is several orders of magnitude greater than traditional optimization-based registration methods (ORs). A major drawback, however, of DLRs is a disregard for the target-pair-specific optimization that is inherent in ORs and instead they rely on a globally optimized network that is trained with a set of training samples to achieve faster registration. Thus, DLRs inherently have degraded ability to adapt to appearance variations and perform poorly, compared to ORs, when image pairs (fixed/moving images) have large differences in appearance. Hence we propose an Appearance Adjustment Network (AAN) where we leverage anatomy edges, through an anatomy-constrained loss function, to generate an anatomy-preserving appearance transformation. We designed the AAN so it can be readily inserted into a wide range of DLRs, to reduce the appearance differences between the fixed and moving images. Our AAN and DLRs’ network can be trained cooperatively in an unsupervised and end-to-end manner.

## Workflow
![workflow](https://github.com/MungoMeng/Image-Registration-AAN/blob/master/Figure/workflow.png)

## Publication
**Coming soon.**

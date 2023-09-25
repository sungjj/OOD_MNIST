# OOD_MNIST
[Out-of-Distribution Detection with Semantic
Mismatch under Masking by Yijun Yang, Ruiyuan Gao, and Qiang Xu](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840369.pdf)

This project performs Out-Of-Distribution (OOD) detection utilizing the main method described in the above paper.

## Dataset
MNIST

In-distribution : 1~9

Out-of-distribution : 0

## Train process

![image](https://github.com/sungjj/OOD_MNIST/assets/136042172/7bcbea6e-8fe2-4497-8408-44c44a91a11a)


## OOD detection Result
zero accuracy : 97.1178

else accuracy : 97.0950,


## Domain Adaption Result
CT -> MR -> CT

![image](https://github.com/sungjj/Volumetric-Unsupervised-Domain-Adaptation-for-Medical-Image-Segmentation/assets/136042172/a3034672-9631-431a-8d67-4c5f331a60fd)

MR -> CT -> MR

![image](https://github.com/sungjj/Volumetric-Unsupervised-Domain-Adaptation-for-Medical-Image-Segmentation/assets/136042172/0d034bcc-18cc-4921-aecb-0f9b4f847320)


## Reference

The code is partially borrowed by [MOODCat]([https://github.com/Seung-Hun-Lee/DRANet](https://github.com/cure-lab/MOODCat))

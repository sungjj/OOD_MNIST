# OOD_MNIST
[Out-of-Distribution Detection with Semantic
Mismatch under Masking by Yijun Yang, Ruiyuan Gao, and Qiang Xu](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840369.pdf)

This project performs Out-Of-Distribution (OOD) detection utilizing the main method described in the above paper.

The current goal is to enhance this method through additional research.

## Dataset
MNIST

In-distribution : 1~9

Out-of-distribution : 0

## Train process

![image](https://github.com/sungjj/OOD_MNIST/assets/136042172/7bcbea6e-8fe2-4497-8408-44c44a91a11a)


## OOD detection Result
alpha: 1.5 beta: 0.95

zero accuracy : 97.1178

else accuracy : 97.0950,


## Points that can be improved
![image](https://github.com/sungjj/OOD_MNIST/assets/136042172/ef4ce5c8-f481-4580-99eb-4c31ad44d74a)

The latent vector of VAE can be used for classification

![image](https://github.com/sungjj/OOD_MNIST/assets/136042172/f353dee9-05ff-4da7-b335-89322bbd329d)

After the classification, we can discriminate whether the image corresponds to the label through a conditional discriminator. If it doesn’t, it is classified as out-of-distribution data.

Through this process, out-of-distribution data can be filtered out in two stages.

## Reference

The code is partially borrowed by [MOODCat](https://github.com/cure-lab/MOODCat)

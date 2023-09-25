SAVE_ROOT=/home/compu/SJJ/OSR/CVAE/0925_2205
GPU=0
#1. train CycleGAN Network
CUDA_VISIBLE_DEVICES=${GPU} python3 test_ood.py \
  --batch_size 1500 \
  --num_classes 9\
  --n_channels 1\
  --img_size 32\
  --latent_vector 80\
  --rec_dir /home/compu/SJJ/OSR/CVAE/0925/best_rec_model.pt\
  --dis_dir /home/compu/SJJ/OSR/CVAE/0925/best_dis_model.pt\
  --cls_dir /home/compu/SJJ/OSR/CVAE/0925_2205/best_cls_model.pt\
  --alphas 1.5 2 2.5\
  --betas 0.90 0.95 0.99
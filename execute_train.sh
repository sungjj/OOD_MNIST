SAVE_ROOT=/home/compu/SJJ/OSR/CVAE/0925
GPU=0
#1. train CycleGAN Network
CUDA_VISIBLE_DEVICES=${GPU} python3 train_with_cls.py \
  --batch_size 1500 \
  --max_epochs 20000 \
  --lr 0.0002 \
  --num_classes 9\
  --img_size 32 \
  --save_dir ${SAVE_ROOT}\
  --n_channels 1\
  --latent_vector 80\
  --save_model 500
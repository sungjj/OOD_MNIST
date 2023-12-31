import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys


import numpy as np
import torch
from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn

import random
import warnings
warnings.filterwarnings("ignore")


class custom_MNIST(torch.utils.data.Dataset):
    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(self, root = '/home/compu/SJJ/OSR/', train = True, transform=None):
        super().__init__()
        self.root = root  
        self.train = train  # training set(True) or test set(False)
        self.data, self.targets = self._load_data()
        self.transform = transform

        if self.train: 
          self._del_zeros()
        # else:
        #   self._del_others()
        self._process_labels()

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.root, image_file))
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.root, label_file))
        return data, targets
    
    def _del_zeros(self):
        idx = self.targets != 0
        self.data = self.data[idx]
        self.targets = self.targets[idx]
    
    def _del_others(self):
        idx = self.targets == 0
        self.data = self.data[idx]
        self.targets = self.targets[idx]
    
    def _process_labels(self):
        self.targets -= 1  # 라벨을 1씩 감소시킴
        self.targets[self.targets == -1] = 9  # 0 라벨을 9로 변경
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy()
        if self.transform is not None:
          img = self.transform(img)

        ###############################
        #                             #
        #     You can modify here     #
        #                             #
        ###############################

        return img, target

    def __len__(self):
        return len(self.data)
    
    
def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Resize((32,32))])

    total_dataset = custom_MNIST(train=True, transform=transform)
    train_size = int(0.9 * len(total_dataset))
    valid_size = len(total_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(total_dataset, [train_size, valid_size])
    test_dataset = custom_MNIST(train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1500, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1500, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1500, shuffle=True)
    
    return train_loader, test_loader, valid_loader
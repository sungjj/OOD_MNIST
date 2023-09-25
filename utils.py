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

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


import random
import warnings
warnings.filterwarnings("ignore")


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (x, y) in enumerate(data_loader):
            
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += y.size(0)
        correct_pred += (predicted_labels == y).sum()
    return correct_pred.float()/num_examples * 100

class Meter:
    def __init__(self):
        self.list = []
    def update(self, item):
        self.list.append(item)
    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

def compute_accuracy_ood(values,output, labels, cls_pred, alpha, beta):
    condition = torch.logical_or(output < alpha, values < beta) #output이 dis loss고 value가 확률
    pred = torch.where(condition, torch.tensor(9).to(output.device), torch.tensor(cls_pred).to(output.device))
    zero_idx= ((labels==9).nonzero(as_tuple=True)[0])
    else_idx= ((labels!=9).nonzero(as_tuple=True)[0])
    #print(else_idx)
    #   else_acc = (pred[else_idx] == labels[else_idx]).type(torch.float).mean().item()*100
    zero_acc = (pred[zero_idx] == labels[zero_idx]).type(torch.float).mean().item()*100
    else_acc = (pred[else_idx] == labels[else_idx]).type(torch.float).mean().item()*100
    #print(else_acc)

    return zero_acc, else_acc

def save_model(enc_model,cls_model,dec_model,dis_model,epoch,save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '{}/model_{}.pt'.format(save_dir, epoch)
    torch.save({'enc_model':enc_model.state_dict(),
                'cls_model':cls_model.state_dict(),
                'dec_model':dec_model.state_dict(),
                'dis_model':dis_model.state_dict(),
                },file_name)


def plot(epoch, pred, recon_loss, y,x,name='train_', save_dir='.'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(6,2,2*i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(recon_loss[i]))
    for i in range(6):
        ax = fig.add_subplot(6,2,2*i+2)
        ax.imshow(x[i,0],cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(y[i]+1))
    plt.savefig("{}/{}epoch_{}.jpg".format(save_dir,name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar):
    # print(x.device,'x')
    # print(pred.device,'pred')
    # print(mu.device,'mu')
    # print(logvar.device,'logvar')
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld


def loss_hinge_dis(dis_fake, dis_real, weight_real = None, weight_fake = None):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real+loss_fake
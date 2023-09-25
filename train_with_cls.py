import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys

import argparse


import numpy as np
import torch
from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn


from utils import *
from models import *
from data import *
from models_ops import *

import random
import warnings
warnings.filterwarnings("ignore")


def train(epoch, enc_model, cls_model, dec_model, dis_model,train_loader, enc_opt, dec_opt, dis_opt, cls_opt, save_dir):
    reconstruction_loss = 0
    kld_loss = 0
    total_rec_loss = 0
    total_cls_loss = 0
    total_dis_loss = 0
    criterion = nn.CrossEntropyLoss()
    for i,(x,y) in enumerate(train_loader):
            label = np.zeros((x.shape[0], 9))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)
            
            
            Int_Modi = random.randint(1, 8)
            wrong_label = ((y + Int_Modi) % 9)
            wrong_label_one_hot = np.zeros((x.shape[0], 9))
            wrong_label_one_hot[np.arange(x.shape[0]), wrong_label] = 1
            wrong_label_tensor = torch.tensor(wrong_label_one_hot)
            
            wrong_label_tensor=wrong_label_tensor.to(device)
            x=x.to(device)
            y=y.to(device)
            label=label.to(device)
            wrong_label=wrong_label.to(device)
            enc_opt.zero_grad()
            dec_opt.zero_grad()   
            z, mu, logvar = enc_model(x)
            #classification network training
            # if epoch>0:
            #     enc_model.eval()
            #     dec_model.eval()
            #     dis_model.eval()
            #     logit=cls_model(z.detach())
            #     #print(logit)
            #     cls_loss=criterion(logit,y)
            #     print(cls_loss)
            #     print(logit[0])
            #     print(y[0])
            #     total_cls_loss+=cls_loss
            #     cls_loss.backward()
            #     cls_opt.step()
            #     #print('cls loss:',cls_loss)
            #     enc_model.train()
            #     dec_model.train()
            #     dis_model.train()
            
            pred=dec_model(z,label)
            recon_loss, kld = loss_function(x.cuda(),pred.cuda(), mu.cuda(), logvar.cuda())
            rec_loss = recon_loss + kld
            rec_loss.backward()
            enc_opt.step()
            dec_opt.step()
            #conditional discriminator training
            if (epoch>50 and epoch%10==0):
                enc_model.eval()
                dec_model.eval()
                cls_model.eval()
                dis_opt.zero_grad()
                z, _, _ = enc_model(x)
                pred=dec_model(z,label)
                sim_tensor = torch.cat((x, pred.detach()), dim=1)
                wrong_label_pred = dec_model(z,wrong_label_tensor)
                dif_tensor = torch.cat((x,wrong_label_pred.detach()), dim=1)
                output=dis_model(sim_tensor, y)
                wrong_label_output=dis_model(dif_tensor,wrong_label)
                dis_loss=loss_hinge_dis(wrong_label_output,output)
                total_dis_loss+=dis_loss
                dis_loss.backward()
                dis_opt.step()
                #print(dis_loss)
                enc_model.train()
                dec_model.train()
                cls_model.train()


            total_rec_loss += rec_loss
            reconstruction_loss += recon_loss
            kld_loss += kld
            # if i == 0 and epoch%50 == 0:
            #     print("Gradients")
            #     for name,param in enc_model.named_parameters():
            #         if "bias" in name:
            #             print(name,param.grad[0],end=" ")
            #         else:
            #             print(name,param.grad[0,0],end=" ")
            if i==0:
                recon_losses = []
                #ssim_losses =[]
                for k in range(6):
                    recon_loss_6 = F.mse_loss(x[k].to(device),pred[k].to(device),reduction='mean')  # 또는 'sum'
                    recon_losses.append(recon_loss_6.item())  # 손실 값을 리스트에 추가
                    x_np = x[k].cpu().detach().numpy()  # PyTorch 텐서를 NumPy 배열로 변환
                    pred_np = pred[k].cpu().detach().numpy()  # PyTorch 텐서를 NumPy 배열로 변환
                    
                    #ssim_loss_6 = structural_similarity(x_np, pred_np, multichannel=False)
                    #ssim_losses.append(ssim_loss_6)
                    #print(ssim_losses)
                #print(recon_losses)
                plot(epoch, pred.cpu().data.numpy(),recon_losses , y.cpu().data.numpy(), x.cpu().data.numpy(),name='train_', save_dir=save_dir)

    total_cls_loss /= len(train_loader.dataset)
    total_dis_loss /= len(train_loader.dataset)
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_rec_loss /= len(train_loader.dataset)
    return total_rec_loss, total_cls_loss, total_dis_loss

def test(epoch, enc_model, cls_model, dec_model, dis_model, data_loader, save_dir):
    total_eval_rec_loss=0
    total_eval_cls_loss=0
    total_eval_dis_loss=0
    criterion = nn.MultiLabelSoftMarginLoss()
    with torch.no_grad():
        for i,(x,y) in enumerate(data_loader):
                label = np.zeros((x.shape[0], 9))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)
                
                
                Int_Modi = random.randint(1, 8)
                wrong_label = ((y + Int_Modi) % 9)
                wrong_label_one_hot = np.zeros((x.shape[0], 9))
                wrong_label_one_hot[np.arange(x.shape[0]), wrong_label] = 1
                wrong_label_tensor = torch.tensor(wrong_label_one_hot)
                
                wrong_label_tensor=wrong_label_tensor.to(device)
                x=x.to(device)
                y=y.to(device)
                label=label.to(device)
                wrong_label=wrong_label.to(device)
                
                z, mu, logvar = enc_model(x)
                logit=cls_model(z.detach())
                cls_loss=criterion(logit,label)
                
                total_eval_cls_loss+=cls_loss
                pred=dec_model(z,label)
                recon_loss, kld = loss_function(x.cuda(),pred.cuda(), mu.cuda(), logvar.cuda())

                rec_loss = recon_loss + kld
                total_eval_rec_loss += rec_loss
                
                pred=dec_model(z,label)
                sim_tensor = torch.cat((x, pred), dim=1)
                wrong_label_pred = dec_model(z,wrong_label_tensor)
                dif_tensor = torch.cat((x,wrong_label_pred), dim=1)
                output=dis_model(sim_tensor, y)
                wrong_label_output=dis_model(dif_tensor,wrong_label)
                dis_loss=loss_hinge_dis(wrong_label_output,output)
                total_eval_dis_loss+=dis_loss
                

                if i==0:
                    recon_losses = []
                    #ssim_losses =[]
                    for k in range(6):
                        recon_loss_6 = F.mse_loss(x[k].to(device),pred[k].to(device),reduction='mean')  # 또는 'sum'
                        recon_losses.append(recon_loss_6.item())  # 손실 값을 리스트에 추가
                        x_np = x[k].cpu().detach().numpy()  # PyTorch 텐서를 NumPy 배열로 변환
                        pred_np = pred[k].cpu().detach().numpy()  # PyTorch 텐서를 NumPy 배열로 변환
                        
                        #ssim_loss_6 = structural_similarity(x_np, pred_np, multichannel=False)
                        #ssim_losses.append(ssim_loss_6)
                        #print(ssim_losses)
                    #print(recon_losses)
                    plot(epoch, pred.cpu().data.numpy(),recon_losses , y.cpu().data.numpy(), x.cpu().data.numpy(),name='valid_', save_dir=save_dir)


    total_eval_rec_loss /= len(test_loader.dataset)
    total_eval_cls_loss /= len(test_loader.dataset)
    total_eval_dis_loss  /= len(test_loader.dataset)
    return total_eval_rec_loss, total_eval_cls_loss*10000, total_eval_dis_loss*10000





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1500, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--max_epochs", default=20000, type=int)
    parser.add_argument("--num_classes", default=9, type=int)
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--save_dir", default='/home/compu/SJJ/CycleGAN_seg_2.5d/SDCUDA_result_0925', help='directory to save images', type=str)
    parser.add_argument("--n_channels", default=1,type=int )
    parser.add_argument("--latent_vector", default=80,type=int )
    parser.add_argument("--save_model", default=500,help='Set the interval for saving model parameters in terms of the number of iterations',type=int )
    args = parser.parse_args()
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    load_epoch=0
    batch_size = args.batch_size
    learning_rate = args.lr
    max_epoch = args.max_epochs
    device = torch.device("cuda")
    num_workers = 5
    generate = True
    train_loader, test_loader, valid_loader = load_data()
    print("dataloader created")
    enc_model = Encoder(isize=args.img_size, nz=args.latent_vector, nc=1, ndf=64).to(device)
    cls_model = SvmClassifier_deep(input_size=args.latent_vector,num_classes=args.num_classes).to(device)
    dec_model = Decoder(latent_size=args.latent_vector, num_classes=args.num_classes).to(device)
    dis_model=Discriminator(img_size=args.img_size, num_classes=args.num_classes, initialize=True, input_dim=2, d_conv_dim=64).to(device)
    print("model created")
    
    # state_dict_rec=torch.load('/home/compu/SJJ/OSR/CVAE/0918_1509/best_rec_model.pt')
    # enc_model.load_state_dict(state_dict_rec['enc_model'])
    # dec_model.load_state_dict(state_dict_rec['dec_model'])
    # # state_dict_cls=torch.load('/home/compu/SJJ/OSR/CVAE/0918_1509/best_cls_model.pt')
    # # cls_model.load_state_dict(state_dict_cls['cls_model'])
    # state_dict_dis=torch.load('/home/compu/SJJ/OSR/CVAE/0918_1509/best_dis_model.pt')
    # dis_model.load_state_dict(state_dict_dis['dis_model'])
    
    # if load_epoch > 0:
    #     enc_model.load_state_dict(torch.load('/home/compu/SJJ/OSR/CVAE/checkpoints_0914/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
    #     print("model {} loaded".format(load_epoch))

    enc_optimizer = optim.Adam(enc_model.parameters(), lr=learning_rate, betas=(0.5,0.999), eps=1e-6)
    dec_optimizer = optim.Adam(dec_model.parameters(), lr=learning_rate, betas=(0.5,0.999), eps=1e-6)
    dis_optimizer = optim.Adam(dis_model.parameters(), lr=learning_rate, betas=(0.5,0.999), eps=1e-6)
    cls_optimizer = optim.Adam(cls_model.parameters(),lr=learning_rate, betas=(0.5,0.999), eps=1e-6)


    train_loss_list = []
    test_loss_list = []
    best_eval_rec_loss=99999999
    best_eval_cls_loss=99999999
    best_eval_dis_loss=99999999

    for i in range(load_epoch+1, max_epoch):
        enc_model.train()
        cls_model.train()
        dec_model.train()
        dis_model.train()
        total_rec_loss, total_cls_loss, total_dis_loss = train(i, enc_model, cls_model, dec_model, dis_model, train_loader, enc_optimizer, dec_optimizer, dis_optimizer, cls_optimizer, save_dir)
        
        with torch.no_grad():
            enc_model.eval()
            cls_model.eval()
            dec_model.eval()
            dis_model.eval()
            eval_rec_loss, eval_cls_loss, eval_dis_loss = test(i, enc_model, cls_model, dec_model, dis_model, valid_loader, save_dir)
            if i>100:
                if eval_rec_loss<best_eval_rec_loss:
                    file_name = '{}/best_rec_model.pt'.format(save_dir)
                    torch.save({'enc_model':enc_model.state_dict(),
                                'dec_model':dec_model.state_dict(),
                                },file_name)
                    best_eval_rec_loss=eval_rec_loss
                    print('best rec loss has changed!')
                if eval_cls_loss<best_eval_cls_loss:
                    file_name = '{}/best_cls_model.pt'.format(save_dir)
                    torch.save({'cls_model':cls_model.state_dict(),
                                },file_name)
                    best_eval_cls_loss=eval_cls_loss
                    print('best cls loss has changed!')
                if eval_dis_loss<best_eval_dis_loss:
                    file_name = '{}/best_dis_model.pt'.format(save_dir)
                    torch.save({'dis_model':dis_model.state_dict(),
                                },file_name)                            
                    best_eval_dis_loss=eval_dis_loss
                    print('best dis loss has changed!')
            # if generate:
            #     z = torch.randn(6, 32).to(device)
            #     y = torch.tensor([1,2,3,4,5,6]) - 1
            #     generate_image(i,z, y, model)
            
        print("Epoch: {}/{} Recon loss: {}, Classification loss: {}, Discrimination loss:{}".format(i, max_epoch, total_rec_loss, total_cls_loss, total_dis_loss))
        # print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))
        if i%args.save_model==0:
            save_model(enc_model,cls_model,dec_model,dis_model, i, save_dir)
import os

from utils import *
from models import *
from data import *
from models_ops import *

import argparse
import random
import warnings
warnings.filterwarnings("ignore")


def train(train_loader, model, optimizer, epoch, max_epochs):
   criterion = nn.CrossEntropyLoss()
   for batch_idx, (x, y) in enumerate(train_loader): 
        # label = np.zeros((x.shape[0], 9))
        # label[np.arange(x.shape[0]), y] = 1
        # label = torch.tensor(label)
        
        x = x.to(device)
        y = y.to(device)
        #label= label.to(device)
        
        logits = model(x)
        
        #loss = F.multilabel_soft_margin_loss(logits, label)
        loss=criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                   %(epoch+1, max_epochs, batch_idx, 
                     len(train_loader), loss))
            
def eval(valid_loader,model,epoch,max_epochs,best_accuracy,save_dir):
    with torch.set_grad_enabled(False): # save memory during inference
        accuracy =  compute_accuracy(model, valid_loader, device=device)
        print('Epoch: %03d/%03d | Test: %.3f%%' % (
              epoch+1, max_epochs, 
              accuracy))
        if accuracy > best_accuracy:
            file_name = '{}/best_cls_model.pt'.format(save_dir)
            torch.save(model.state_dict(),file_name)
            best_accuracy=accuracy
            print('The best model has changed.')
        return best_accuracy
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1500, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_epochs", default=20000, type=int)
    parser.add_argument("--num_classes", default=9, type=int)
    parser.add_argument("--save_dir", default='/home/compu/SJJ/CycleGAN_seg_2.5d/SDCUDA_result_0925', help='directory to save images', type=str)
    parser.add_argument("--n_channels", default=1,type=int )
    args = parser.parse_args()
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    device = torch.device("cuda")
    
    train_loader, test_loader, valid_loader = load_data()
    
    model=resnet18(args.num_classes)
    model.to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy=0
    for epoch in range(args.max_epochs):
        model.train()
        train(train_loader=train_loader, model=model, optimizer=optimizer, epoch=epoch, max_epochs=args.max_epochs)
        model.eval()
        best_accuracy=eval(valid_loader=valid_loader,model=model,epoch=epoch,max_epochs=args.max_epochs,best_accuracy=best_accuracy,save_dir=args.save_dir)

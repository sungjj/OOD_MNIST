from utils import *
from models import *
from data import *
from models_ops import *
import argparse
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1500, type=int)
    parser.add_argument("--num_classes", default=9, type=int)
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--n_channels", default=1,type=int )
    parser.add_argument("--latent_vector", default=80,type=int )
    parser.add_argument("--save_model", default=500,help='Set the interval for saving model parameters in terms of the number of iterations',type=int )
    parser.add_argument("--rec_dir", default='/home/compu/SJJ/OSR/CVAE/0925/best_rec_model.pt',type=str )
    parser.add_argument("--dis_dir", default='/home/compu/SJJ/OSR/CVAE/0925/best_dis_model.pt',type=str )
    parser.add_argument("--cls_dir", default='/home/compu/SJJ/OSR/CVAE/0925_2205/best_cls_model.pt',type=str )
    parser.add_argument('--alphas', nargs='+', type=float, default=[1,2,3], help='List of alphas')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.97,0.98,0.99], help='List of betas')


    args = parser.parse_args()
    
    device=torch.device('cuda')
    
    args = parser.parse_args()

    
    train_loader, test_loader, valid_loader = load_data()
    print("dataloader created")
    acc_zero = Meter()
    acc_else = Meter()
    enc_model = Encoder(isize=args.img_size, nz=args.latent_vector, nc=args.n_channels, ndf=64).to(device)
    cls_model = resnet18(args.num_classes).to(device)
    dec_model = Decoder(latent_size=args.latent_vector, num_classes=args.num_classes).to(device)
    dis_model=Discriminator(img_size=args.img_size, num_classes=args.num_classes, initialize=True, input_dim=2, d_conv_dim=64).to(device)
    print("model created")
    state_dict_rec=torch.load(args.rec_dir)
    enc_model.load_state_dict(state_dict_rec['enc_model'])
    dec_model.load_state_dict(state_dict_rec['dec_model'])
    cls_model.load_state_dict(torch.load(args.cls_dir))
    state_dict_dis=torch.load(args.dis_dir)
    dis_model.load_state_dict(state_dict_dis['dis_model'])
    print("model loaded")
    for alpha in args.alphas:
        for beta in args.betas:
            with torch.no_grad():
                enc_model.eval()
                cls_model.eval()
                dec_model.eval()
                dis_model.eval()
                for i, (x, y) in enumerate(test_loader):
                    
                    x_c=x.to(device)
                    #y_c=y.to(device)
                    # label=label.to(device)
                    # wrong_label=wrong_label.to(device)
                    # wrong_label_tensor=wrong_label_tensor.to(device)
                    
                    logits= cls_model(x_c)
                    probabilities = F.softmax(logits, dim=1)
                    #print(probas)
                    values, cls_pred = torch.max(probabilities, dim=1)
                    
                    pred_np = cls_pred.cpu().numpy()
                    
                    label = np.zeros((x.shape[0], 9))
                    label[np.arange(x.shape[0]), pred_np] = 1
                    label = torch.tensor(label)
                    label=label.to(device)
                    x=x.to(device)
                    y=y.to(device)
                    
                    z, mu, logvar = enc_model(x)
                    pred=dec_model(z.detach(),label)
                    sim_tensor = torch.cat((x, pred.detach()), dim=1)
                    output=dis_model(sim_tensor, cls_pred)
                    
                    #print('right:',output)
                    #print(y)
                    zero_acc, else_acc = compute_accuracy_ood(values,output, y, cls_pred, alpha, beta)
                    
                    acc_else.update(else_acc)
                    acc_zero.update(zero_acc)

            print("zero_acc: {:.4f}, else_acc: {:.4f}, alpha: {}, beta: {}".format(acc_zero.avg(), acc_else.avg(), alpha, beta))
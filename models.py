import torch 
import torch.nn as nn
import torch.nn.functional as F

from models_ops import *

import warnings
warnings.filterwarnings("ignore")


# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 30

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 9

# Other
DEVICE = "cuda:1"
GRAYSCALE = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        #print(logits.size())
        #logits= F.sigmoid(logits)
        #print(probas)
        return logits



def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()

        self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.bn0 = batchnorm_2d(in_features=in_channels)
        self.bn1 = batchnorm_2d(in_features=out_channels)

        self.activation = nn.ReLU(inplace=True)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscBlock, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if self.ch_mismatch or downsample:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if self.ch_mismatch or downsample:
            self.bn0 = batchnorm_2d(in_features=in_channels)
        self.bn1 = batchnorm_2d(in_features=in_channels)
        self.bn2 = batchnorm_2d(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size,  num_classes, initialize, input_dim=2, d_conv_dim=64):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {"32": [input_dim] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                "64": [input_dim] + [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8],
                                "128": [input_dim] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "256": [input_dim] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
                                "512": [input_dim] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

        d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                 "64": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

        d_down = {"32": [True, True, False, False],
                  "64": [True, True, True, True, False],
                  "128": [True, True, True, True, True, False],
                  "256": [True, True, True, True, True, True, False],
                  "512": [True, True, True, True, True, True, True, False]}

        self.in_dims  = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
                                              out_channels=self.out_dims[index],
                                              )]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                           out_channels=self.out_dims[index],
                                           downsample=down[index])]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = nn.ReLU(inplace=True)

        self.linear1 = linear(in_features=self.out_dims[-1], out_features=1)

        self.embedding = embedding(num_classes, self.out_dims[-1])

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, x, label, evaluation=False):
        # with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = self.activation(h)
        h = torch.sum(h, dim=[2,3])

        authen_output = torch.squeeze(self.linear1(h))
        #proj = torch.sum(torch.mul(self.embedding(label), h), 1)

        return  authen_output

class SvmClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SvmClassifier, self).__init__()
        self.layer1=nn.Linear(input_size, input_size//2)
        self.layer2=nn.Linear(input_size//2, input_size//4)
        self.fc=nn.Linear(input_size//4, num_classes)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.fc(x)
        return x
    
    
class SvmClassifier_deep(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SvmClassifier_deep, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.layer1=nn.Linear(input_size, 64)
        self.bn1 = batchnorm_2d(in_features=64)
        self.layer2=nn.Linear(64,128)
        self.bn1 = batchnorm_2d(in_features=128)
        self.fc=nn.Linear(128,num_classes)
    def forward(self,x):
        x=self.layer1(x)
        #x=self.bn1(x)
        x=self.activation(x)
        x=self.layer2(x)
        #x=self.bn2(x)
        x=self.activation(x)
        x=self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, add_final_conv=True):
        super(Encoder, self).__init__()
        self.nz = nz
        encoder = nn.Sequential()
        encoder.add_module('initial-conv-{0}-{1}'.format(nc, ndf), 
                            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        encoder.add_module('initial-relu-{0}'.format(ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize/2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            encoder.add_module('pyramid-{0}-{1}'.format(in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)
                )
            encoder.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                nn.BatchNorm2d(out_feat))
            encoder.add_module('pyramid-{0}-relu'.format(out_feat),
                nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        if add_final_conv:
            encoder.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        self.encoder = encoder
        self.z_mean_calc = nn.Linear(nz, nz)
        self.z_log_var_calc = nn.Linear(nz, nz)

    def forward(self, input):
        output = self.encoder(input)
        z_mean = self.z_mean_calc(output.view(-1, self.nz)).cuda()
        z_log_var = self.z_log_var_calc(output.view(-1, self.nz)).cuda()
  
        z_mean_0 = z_mean
        z_log_var_0 = z_log_var
        epsilon = torch.randn(size=(z_mean_0.view(-1,self.nz).shape[0], self.nz)).cuda()
        latent_i_star = z_mean_0 + torch.exp(z_log_var_0 / 2) * epsilon
  
        z_mean_ret =  z_mean_0
        z_log_var_ret =  z_log_var_0
  
        return z_mean_ret, z_log_var_ret, latent_i_star.type(torch.FloatTensor)
    
class Decoder(nn.Module):
    def __init__(self,latent_size=32,num_classes=9):
        super(Decoder,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,800)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(1, 1, kernel_size=4)
        
    def decoder(self, z):
        t = F.relu(self.linear2(z))
        #print(t.size())
        t = F.relu(self.linear3(t))
        #print(t.size())
        t = self.unFlatten(t)
        #print(t.size())
        t = F.relu(self.conv3(t))
        #print(t.size())
        t = F.relu(self.conv4(t))
        #print(t.size())
        t = F.relu(self.conv5(t))
        #print(t.size())
        return t

    def unFlatten(self, x):
        return x.reshape((x.shape[0], 32, 5, 5))
    
    def forward(self, z, y):
        #print(z.size()) #(10000, 32)
        # Class conditioning at decoder
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred
import torch
import torch.nn as nn
from torch.autograd import Variable
import math  # init
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,v,w,stride,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, v, kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(v)
        self.relu = nn.ReLU(inplace=True)
        inplanes =v
        self.conv2 = nn.Conv2d(inplanes, w, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        # self.residual=
        inplanes = w
        self.downsample = downsample
    def forward(self, x):
        residual = x
        # print("******",residual.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print("......",out.size())
        if self.downsample is not None:
            # if out.size()[1] != residual.size()[1]:
                # residual=nn.Conv2d(residual.size()[1], out.size()[1],kernel_size=1,bias=False)(x)
            residual = self.downsample(x)
                # print("++++++",residual.size())
        out += residual
        out = self.relu(out)
        return out




class vgg(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            # cfg = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
            # cfg = [64,
            #        64,64,
            #        128,128,
            #        256,256,
            #        512,512]
            cfg = [64,
                   64,64,64,64,
                   128,128,128,128,
                   256,256,256,256,
                   512,512,512,512]
            length = len(cfg)
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        inplane = 64
        inplanes = 64
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            downsample=None
            if i==0:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                inplanes = v
            elif i>0:
                i=2*i-1
                if i==1 or i==5 or i==9 or i==13:
                    stride=2
                else:
                    stride=1
                if i<16:
                    if inplanes != cfg[i+1] or stride != 1:
                        downsample = nn.Sequential(
                            nn.Conv2d(inplanes, cfg[i+1], kernel_size=1, stride=stride, padding=0, bias=False),
                            nn.BatchNorm2d(cfg[i+1]))
                    layers.append(BasicBlock(inplanes, cfg[i], cfg[i+1], stride, downsample))
                    inplanes = cfg[i+1]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        # print("<><><>",x.size())
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg()
    print(net)
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)
    total = 0
    layers_params = (net.parameters())
    i = 1
    for layer in layers_params:
        print("per layer", "(", i, ")", list(layer.size()))
        l = np.cumprod(list(layer.size()))[-1]
        total = total + l
        i += 1

    print("parameters =", total)
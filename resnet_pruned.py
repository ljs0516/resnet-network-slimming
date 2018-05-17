# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

# from resnet18_right import vgg
from resnet18_right_half import vgg
import numpy as np

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='pruned_0.5.pth.tar', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


model = vgg()
# original_layers_params = model.state_dict()
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        original_layers_params=model.state_dict()
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



total = 0
flag = False
for k,m in enumerate(model.modules()):
    # print(k,m)
    if k==3 or k==7 or k==16 or k==22 or k==31 or k==37 or k==46 or k==52 or k==61:
        flag=True
    if isinstance(m, nn.BatchNorm2d) and flag:
        total += m.weight.data.shape[0]
        flag = False
flag = False
bn = torch.zeros(total)
index = 0
for k,m in enumerate(model.modules()):
    if k==3 or k==7 or k==16 or k==22 or k==31 or k==37 or k==46 or k==52 or k==61:
        flag=True
    if isinstance(m, nn.BatchNorm2d) and flag:
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size
        flag = False

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]
pruned = 0
cfg = []
cfg_mask = []
flag = False
cfg_flag = False
BN_flag=False
n = -1
for k, m in enumerate(model.modules()):
    if k==3 or k==7 or k==16 or k==22 or k==31 or k==37 or k==46 or k==52 or k==61:
        flag = True  # 1*1卷积后的进行剪枝
        cfg_flag = True
    if k==10 or k==19:
        cfg.append(64)
    if k==25 or k==34:
        cfg.append(128)
    if k==40 or k==49:
        cfg.append(256)
    if k==55or k==64:
        cfg.append(512)
    if isinstance(m, nn.BatchNorm2d) and BN_flag:
        if flag == True:#对1*1进行卷积剪枝
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if cfg_flag == True:
                cfg.append(num)
                cfg_flag = False
            flag = False
            cfg_mask.append(mask.clone())
            n += 1
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], num))
            BN_flag=False
        elif flag == False:#针对的是depthwise,虽然他理论上没有进行bn排序，但是为了保持和上个通道一样，所以也要剪掉
            weight_copy = m.weight.data.clone()
            # if k>6:
            # mask = cfg_mask[n]#针对depthwise层，剪掉的东西和上层的1*1 相同。因此使用cfg_mask[n]
            # else:
            mask = weight_copy.abs().gt(0).float().cuda()#小于6的不用剪掉
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            cfg_mask.append(mask.clone())
            n += 1
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], num))
            BN_flag=False
    elif isinstance(m,nn.Conv2d):
        if m.kernel_size[0]==3:
            BN_flag=True

pruned_ratio = pruned/total
print(cfg)
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

test()


# Make real prune
newmodel = vgg(cfg=cfg)
newmodel.cuda()
print("<><><><><><>",newmodel)
flag=False
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d) and flag:
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        idx1=torch.from_numpy(idx1)
        idx1 = idx1.type(torch.cuda.LongTensor)
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        # print("start_mask is:",start_mask)
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
        flag=False
    elif isinstance(m0, nn.Conv2d):
        if m0.kernel_size[0] == 3:
            flag=True
            # print(flag)
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            # idx1 = torch.from_numpy(idx1)
            # idx1 = idx1.type(torch.cuda.LongTensor)
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            # m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[:, idx0].clone()


torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
# print("aaaa",newmodel)
newparameters=newmodel.state_dict()
# print("pruned parameters are:",Parameters(newmodel))
model = newmodel
total = 0
layers_params = (model.parameters())
i = 1
for layer in layers_params:
    # print("per layer", "(", i, ")", list(layer.size()))
    l = np.cumprod(list(layer.size()))[-1]
    total = total + l
    i += 1

print("parameters =", total)
test()
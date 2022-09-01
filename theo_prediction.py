from __future__ import print_function
import os, os.path, sys, time, math
import numpy as np
import torch
#import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
#import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import ToTensor
import argparse 
from collections import OrderedDict
import torch.nn.init as init
from teachers import mnist_dataset


def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)

def correlation(x, y):
    return torch.dot(x,y)/len(x)

def normalized_dot(x,y):
    return torch.dot(x,y)/(torch.norm(x),torch.norm(y))

def kernel(x,y): 
    if (x == y).all():
        return 1
    else:
        temp = correlation(x,y)/(torch.sqrt(correlation(x,x)*correlation(y,y)))
        return (2/np.pi)*(np.arcsin(temp))

def kernel_tanh(x,y): 
    u = normalized_dot(x,y)

    temp = correlation(x,y)/(torch.sqrt(correlation(x,x)*correlation(y,y)))
    return torch.norm(x)*torch.norm(y)

def kmatrix(data):
    P = len(data)
    K = torch.randn(P,P)
    for i in range(P): 
        for j in range(P):            
            K[i][j] = kernel(data[i], data[j])
    return K

def gen_error_1hl_wrong(data, x,y,labels, N1, wd, lr):
    P = len(data)
    sum1 = 0
    sum2 = 0
    l2reg = wd /lr 
    K = kmatrix(data)
    invK = torch.inverse(K)
    for mu in range(P):
        for nu in range(P):
            sum1 += kernel(x, data[mu])*invK[mu][nu]*labels[mu]
    sum1 = -sum1+ y
    for mu in range(P):
        for nu in range(P):
            sum2 += kernel(x, data[mu])*invK[mu][nu]*kernel(x, data[nu])
    sum2 = -sum2 + kernel(x,x)
    Qbar = qbar(labels, invK, N1)
    return sum1**2 - (Qbar/l2reg)*sum2


def gen_error_1hl(data, x,y,labels, wd, lr, invK, Qbar):
    P = len(data)
    sum1 = 0
    sum2 = 0
    l2reg = wd /lr 
    K0 = torch.tensor([kernel(x, data[mu]) for mu in range(P)])
    K0_invK = torch.matmul(K0, invK)
    sum1 = -torch.dot(K0_invK, labels.type(torch.FloatTensor)) + y
    sum2 = -torch.dot(K0_invK, K0) + kernel(x,x)
    #Qbar = qbar(labels, invK, N1)
    return sum1**2 - (Qbar/l2reg)*sum2

def qbar(labels, invK, N1):
    labels = labels.type(torch.DoubleTensor)
    invK = invK.type(torch.DoubleTensor)
    P = len(labels)
    alpha1 = P/N1
    yky = torch.matmul(torch.matmul(torch.t(labels), invK), labels)
    return ((alpha1-1)-torch.sqrt((alpha1-1)**2 + 4*alpha1*yky/P))/2
        




N0 = 196
N1 = 500
wd = 1e-05
lr = 0.01
wd = 1.
lr = 1.
P_list = [int(69*(1.6**i)) for i in range(9)]

teacher_class = mnist_dataset(N0)
teacher_class.batch_size = 1
_,testloader,_,_ = teacher_class.make_data(1,1, False)
dataiter = iter(testloader)
x, y = next(dataiter)
y = y.item()
x = x.flatten(start_dim = 0)



for p in P_list:
    _,_,data,labels = teacher_class.make_data(p,1, False)
    data = data.flatten(start_dim = 1)
    K = kmatrix(data)
    invK = torch.inverse(K)
    for N1 in [1000, 1500, 2000, 5000]:
        f = open(f"theory_mnist_N1_{N1}.txt", "a")
        Qbar = qbar(labels, invK, N1)
        print(p,gen_error_1hl(data, x,y,labels, wd, lr, invK, Qbar).item()/teacher_class.trivial_predictor(1), file = f)
        f.close








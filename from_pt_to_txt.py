from __future__ import print_function
import os, os.path, sys, time, math
import numpy as np
import torch

import torch.nn as nn



import torchvision.models as models

import argparse 


from teachers import mnist_dataset



def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)



N1 = 500
N0 = 500
lr = 0.1
wd = 0.0
noise = 0.0 
bs = 50
lambda0 = 1.0
lambda1 = 1.0
teacher_type = "linear"
net_type = "1hl"
opt = "sgd"
bias = "True"
device = "cuda"
Ptest = 1000
P = 723

mother_dir = './runs_erf/'
first_subdir = mother_dir + f'teacher_{teacher_type}_net_{net_type}_opt_{opt}_bias_{bias}/'


#attributes_string = f'lr_{lr}_w_decay_{wd}_noise_{noise}_bs_{bs}_lambda0_{lambda0}_lambda1_{lambda1}_N_{N0}_N1_{N1}'
#run_folder = first_subdir + attributes_string + attributes_string()

trainsetFilename = f'{first_subdir}trainset_N_{N0}_P_{P}_Ptest_{Ptest}.pt'

loaded = torch.load(trainsetFilename, map_location=torch.device(device))
inputs, targets, test_inputs, test_targets = loaded['inputs'], loaded['targets'], loaded['test_inputs'],loaded['test_targets']

print("\ntrainset was loaded from checkpoint")
outFilename = f'{first_subdir}Trainset_N_{N0}_P_{P}_Ptest_{Ptest}.txt'

f = open(outFilename, "a")

#for t in range(len(test_inputs)):
#    print([[test_inputs[t][i].item() for i in range(N0)], test_targets[t].item()], file = f)


for t in range(len(inputs)):
    print([[inputs[t][i].item() for i in range(N0)], targets[t].item()], file = f)


f.close()












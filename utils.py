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





def load_datasets(dataFilename, labelsFilename, batch_size_train, teacher_class, device):
	try:
		data = torch.load(dataFilename, map_location=torch.device(device))
		labels = torch.load(labelsFilename, map_location=torch.device(device))
		trainset = list(zip(data,labels))
		trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train, shuffle = True)
		_, testloader, _ , _  = teacher_class.make_data()
		print("\ndata was loaded from checkpoint") 
	except:
		print("\ndidn't find data to load")
		trainloader, testloader, data, labels = teacher_class.make_data()
		torch.save(data, dataFilename)
		torch.save(labels, labelsFilename)
	return trainloader, testloader

def load_net_state(solutionFilename, net, device):
    try:
        if os.path.exists(solutionFilename):
            print("\nresuming training..")
        loaded_data = torch.load(solutionFilename, map_location=torch.device(device))
        state_dict = loaded_data['net']
        start_epoch = loaded_data['epoch']
       # start_epoch += 1
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if device == 'cuda':
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.` when you are using cpu you don't need this line
                else:
                    name = k
                    new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)	
        net.to(device)
        #net.train()	
    except: 
        print("\nI didin't find a checkpoint to resume training. I will start training from scratch")	
        cuda_init(net, device)
        init_network(net)
        normalise(net)
        start_epoch = 0

    return net, start_epoch	


#@profile
def save_state(net, epoch, solutionFilename):
	print('Saving at epoch', epoch)
	state = {
	'net': net.state_dict(),
	'epoch': epoch,
	}
	torch.save(state, solutionFilename)
	#print_stats()

#@profile
def cuda_init(net, device):
	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
		CUDA_LAUNCH_BLOCKING=1
	#print_stats()

#@profile
def init_network(net):
    for m in net.modules():
        if isinstance(m,nn.Linear):
            init.normal_(m.weight,std=1/np.sqrt(len(m.weight[0])))
            init.constant_(m.bias,0)
    #print_stats()

#@profile
def normalise(net):
    for m in net.modules():
        if isinstance(m, nn.Linear): 
            with torch.no_grad():
                #True
                #print("i divide by:", len(m.weight[0]))  		
                m.weight /= np.sqrt(len(m.weight[0]))
                m.weight /= len(m.weight[0])
    #print_stats()
























### PROFILER

#from line_profiler import LineProfiler
#profiler = LineProfiler()

def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)
    return inner

def print_stats():
    profiler.print_stats()


def wrapper(func, args): # with star
    return func(*args)
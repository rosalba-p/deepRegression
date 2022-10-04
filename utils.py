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


def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("teacher_type", help="choose a teacher type: linear, quadratic, 1hl, mnist", type=str)
    parser.add_argument("net_type", help="choose a net type: rfm, 1hl, 2hl, resnet18, vgg11, densenet 121 ", type=str)
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default=1e-04)
    parser.add_argument("-wd", "--wd", help="weight decay", type=float, default=1e-05)
    parser.add_argument("-resume_net",help="resume previous training", type=bool, default=False)
    parser.add_argument("-resume_data", help="resume previous training", type=bool, default=False)    
    #parser.add_argument("-resume_trainset", "--resume_trainset", help="resume previous training", type=bool, default=False)
    parser.add_argument("-device", "--device",  type=str, default="cpu")
    parser.add_argument("-epochs", "--epochs", help="number of train epochs", type = int, default = 10000)
    parser.add_argument("-bs", "--bs", help="batch size train", type=int, default=0)
    #you will trigger a series of experiment with increasing dataset size. at each step Pstart -> Pstart + step. the number of steps is: nPoints 
    parser.add_argument("-Pstart", "--Pstart", help="size of training set", type=int, default=200)
    parser.add_argument("-Pnorm", "--Pnorm",  type=int, default=100000)
    parser.add_argument("-step", "--step", help="step to increase size", type=float, default=10)
    parser.add_argument("-nPoints", "--nPoints", help="number of iterations", type=int, default=50)
    #specify the networks you want to use 
    parser.add_argument("-N", "--N", help="size of input data", type=int, default=300)
    parser.add_argument("-N1", "--N1", help="size of first hidden layer", type=int, default=400)
    parser.add_argument("-N2", "--N2", help="size of second hidden layer", type=int, default=400)
    parser.add_argument("-N1T", "--N1T", help="size of teacher's hidden layer", type=int, default=200)
    parser.add_argument("-Ptest", "--Ptest", help="# examples in test set", type=int, default=10000)
    parser.add_argument("-opt", "--opt", type=str, default="sgd") #or adam
    parser.add_argument("-bias", "--bias", type=bool, default=False)
    # you can specify this index if you want to do more than one run of experiments. by default it is set to 0. 
    parser.add_argument("-R", "--R", help="replica index", type=int, default=1)
    parser.add_argument("-checkpoint", "--checkpoint", help="# epochs checkpoint", type=int, default=1000)
    parser.add_argument("-noise", "--noise", help="signal to noise ratio", type=float, default=0.)
    parser.add_argument("-save_data", "--save_data", type = bool, default= True)
    parser.add_argument("-lambda1", type = float, default= 1.)
    parser.add_argument("-lambda0", type = float, default= 1.)
    parser.add_argument("-compute_theory", type = bool, default= False)
    parser.add_argument("-only_theo", type = bool, default= False)
    parser.add_argument("-minR", type = int, default= 0)
    args = parser.parse_args()
    return args


#@profile
def train(net, trainloader,criterion, optimizer,  device, conv):
	net.train()
	train_loss = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		targets = targets.type(torch.FloatTensor).unsqueeze(1)
		inputs, targets = inputs.to(device), targets.to(device)
		if not conv: 
			inputs = inputs.flatten(start_dim = 1)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	#print_stats()
	return(train_loss/(batch_idx+1))


#@profile
def test(net, testloader,criterion, optimizer,device,conv):
	net.eval()
	test_loss = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
				targets = targets.type(torch.FloatTensor).unsqueeze(1)
				inputs, targets = inputs.to(device), targets.to(device)
				if not conv: 
					inputs = inputs.flatten(start_dim = 1)
				outputs = net(inputs)
				loss = criterion(outputs,targets)
				test_loss += loss.item()
	#print_stats()
	return(test_loss/(batch_idx+1))

def train_synthetic(net,data, labels, criterion, optimizer, device, batch_size):
		net.train()
		train_loss = 0
		P = len(data)
		batch_num = max(int(P/batch_size),1)
		s = np.arange(P)
		np.random.shuffle(s)
		data = data[s]
		labels = labels[s]
		for i in range(batch_num):
			start = i*(batch_size)
			inputs, targets = data[start:start+batch_size].to(device), (labels[start:start+batch_size]).unsqueeze(1).to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
			train_loss += loss.item() 
		return train_loss/batch_num

def test_synthetic(net, test_data, test_labels, criterion, optimizer, device, batch_size):
		net.eval()
		test_loss = 0
		P_test = len(test_data)
		batch_num = max(int(P_test/batch_size),1)
		for i in range(batch_num):
			start = i*(batch_size)
			with torch.no_grad():
					inputs, targets = test_data[start:start+batch_size].to(device), (test_labels[start:start+batch_size]).unsqueeze(1).to(device)
					outputs = net(inputs)
					loss = criterion(outputs, targets)
					test_loss += loss.item()
		return test_loss/batch_num


def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)


#def load_datasets_mnist(dataFilename, labelsFilename, batch_size_train, teacher_class, device):
#    try:
#        if os.path.exists(dataFilename):
#            print("\ntrying loading data..")
#        data = torch.load(dataFilename, map_location=torch.device(device))
#        labels = torch.load(labelsFilename, map_location=torch.device(device))
#        trainset = list(zip(data,labels))
#        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train, shuffle = True)
#        _, testloader, _ , _  = teacher_class.make_data()
#        print("\ndata was loaded from checkpoint") 
#    except:
#        print("\ndidn't find data to load")
#        trainloader, testloader, data, labels = teacher_class.make_data()
#        torch.save(data, dataFilename)
#        torch.save(labels, labelsFilename)
#    return trainloader, testloader

def load_net_state(solutionFilename, net, device):
    try:
        if os.path.exists(solutionFilename):
            print("\ntrying loading net..")
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
        print("\nnet was loaded from checkpoint") 	
    except: 
        print("\ndidin't find a checkpoint to resume training. Starting training from scratch")	
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
def init_network(net,bias):
    for m in net.modules():
        if isinstance(m,nn.Linear):
            init.normal_(m.weight,std=1/np.sqrt(len(m.weight[0])))
            #init.normal_(m.weight,std=1)
            if bias == True:
                init.constant_(m.bias,0)
    #print_stats()

#@profile
def normalise(net, layer, lamb):
    for m in range(len(net)):
        with torch.no_grad():
            if m == layer:
                #True
                #print("i divide by:", len(m.weight[0]))  		
                #m.weight /= np.sqrt(len(m.weight[0]))
                net[m].weight /= np.sqrt(lamb)
                #net[m].weight /= 
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



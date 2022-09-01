import torch 
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import torchvision
import numpy as np



class make_1hl(): 

    def __init__(self, input_size, last_layer_size):
        self.input_size = input_size 
        self.last_layer_size = last_layer_size


    def sequential(self, bias):
        return  nn.Sequential(
            nn.Linear(self.input_size, self.last_layer_size, bias=bias),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.last_layer_size, 1, bias=bias),
        )

    def attributes_string(self): 
        return f"_N_{self.input_size}_N1_{self.last_layer_size}"


class make_2hl: 

    def __init__(self, input_size, hid_layer_size, last_layer_size):
        self.input_size = input_size 
        self.hid_layer_size = hid_layer_size
        self.last_layer_size = last_layer_size


    def sequential(self, bias):
        return nn.Sequential(
            nn.Linear(self.input_size, self.hid_layer_size, bias=bias),
            nn.ReLU(),
            nn.Linear(self.hid_layer_size, self.last_layer_size, bias=bias),
            nn.ReLU(),
            nn.Linear(self.last_layer_size, 1, bias=bias),
        )

    def attributes_string(self): 
        return f"_N_{self.input_size}_N1_{self.hid_layer_size}_N2_{self.last_layer_size}"



class make_densenet121: 

    def __init__(self):
        self.last_layer_size = 1024

    def sequential(self):
        net = models.densenet121(pretrained=False)
        net.classifier = nn.Linear(1024,1)
        for x in net.modules():
            if isinstance(x, nn.AvgPool2d) or isinstance(x, nn.MaxPool2d):
                x.ceil_mode = True
        return net

    def attributes_string(self):
        return ""


class make_vgg11: 

    def __init__(self):
        self.last_layer_size = 4096


    def sequential(self):
        net = models.vgg11(pretrained=False)
        net.classifier[6] = nn.Linear(4096,1)
        for x in net.modules():
            if isinstance(x, nn.AvgPool2d) or isinstance(x, nn.MaxPool2d):
                x.ceil_mode = True
        return net

    def attributes_string(self):
        return ""


class make_resnet18: 

    def __init__(self):
        self.last_layer_size = 512

    def sequential(self):
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(512,1)
        for x in net.modules():
            if isinstance(x, nn.AvgPool2d) or isinstance(x, nn.MaxPool2d):
                x.ceil_mode = True
        return net

    def attributes_string(self):
        return ""


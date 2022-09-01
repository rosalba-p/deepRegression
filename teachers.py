from cgi import test
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
#from operator import itemgetter 



"""# **MAIN**"""


class mnist_dataset: 
    
    #@profile
    def __init__(self,N):
        #self.N = 784
        self.N = N
        #self.conv = bool("NaN")
        self.P = float("NaN")
        self.P_test = float("NaN")
        self.batch_size = float("NaN")


    #@profile
    def make_data(self, P,P_test, conv):
        if conv: 
            self.transform_dataset = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, )), 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),            
            ])
        else: 
            self.transform_dataset = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(size = int(np.sqrt(self.N))),          
            ])

        trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=self.transform_dataset)


        trainset_small = torch.utils.data.Subset(trainset,  list(range(P)))
        data,labels = [], []
        for i in range(P):
            data.append(trainset_small[i][0])
            labels.append(torch.tensor(trainset_small[i][1]))

    
        trainloader = torch.utils.data.DataLoader(
        trainset_small, batch_size = self.batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=self.transform_dataset)
    
        testset_small = torch.utils.data.Subset(testset, list(range(P_test)))

        testloader = torch.utils.data.DataLoader(
        testset_small, batch_size = self.batch_size)


        #return trainloader, testloader,  inputs, targetstrainset_small, testset_small
        return trainloader, testloader,  torch.stack(data), torch.stack(labels)


    def resume_data(self, dataFilename, labelsFilename, device): 
        try:
            data = torch.load(dataFilename, map_location=torch.device(device))
            labels = torch.load(labelsFilename, map_location=torch.device(device))
            trainset = list(zip(data,labels))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batch_size, shuffle = True)
            _, testloader, _ , _  = self.make_data()
            print("\ndataset was loaded from checkpoint") 
        except:
            print("\ndidn't find data to load, creating new dataset")
            trainloader, testloader, data, labels = self.make_data()
            torch.save(data, dataFilename)
            torch.save(labels, labelsFilename)
        return trainloader, testloader

    def training_type(self):
        return "train", "test"

    def trivial_predictor(self, Pnorm):
        return 28.5






class linear_dataset: 

    #@profile
    def __init__(self, teacher_vec):
        self.teacher_vec = teacher_vec

        self.N = len(self.teacher_vec)
        self.save_data = True
        self.resume = bool("NaN")


    #@profile
    def make_data(self,P,P_test,conv, trainsetFilename, device):
        resume_status = False 
        try:
            if self.resume: 
                loaded = torch.load(trainsetFilename, map_location=torch.device(device))
                inputs, targets = loaded['inputs'], loaded['targets'], loaded['trivialpred']
                
                print("\ntrainset was loaded from checkpoint")
                resume_status = True  
            else:
                raise Exception()
        except:
            print("\nCreating new dataset..")
            inputs = torch.randn((P,self.N))
            targets = torch.sum(self.teacher_vec * inputs, dim=1) 
            
        test_inputs = torch.randn((P_test,self.N))
        test_targets = torch.sum(self.teacher_vec * test_inputs, dim=1)

        if conv: 
            inputs, test_inputs = inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3))), test_inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3)))

        return  inputs, targets, test_inputs, test_targets,resume_status


    def training_type(self):
        return "train_synthetic", "test_synthetic"
    
    def trivial_predictor(self, P_norm):

        inputs = torch.randn((P_norm,self.N))
        targets_pred = torch.sum(self.teacher_vec * inputs, dim=1) 
        R0 = 0
        for t in targets_pred:
                R0 += t**2
        return (R0/P_norm).item()


class random_dataset: 

    #@profile
    def __init__(self, N):
        self.N = N


    #@profile
    def make_data(self,P,P_test, conv, trainsetFilename, device):
    
        inputs = torch.randn((P,self.N))
        resumed = False
        targets = torch.randn(P)
        test_inputs = torch.randn((P_test,self.N))
        test_targets = torch.randn(P_test)

        if conv: 
            inputs, test_inputs = inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3))), test_inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3)))

        return inputs, targets, test_inputs, test_targets,resumed

    def training_type(self):
        return "train_synthetic", "test_synthetic"

    def trivial_predictor(self, Pnorm):
        return 1







   
class quadratic_dataset: 

    #@profile
    def __init__(self, N, teacher_vec):
        self.teacher_vec = teacher_vec
        self.N = N

    #@profile
    def make_data(self,P,P_test,conv, trainsetFilename, device):
        resume_status = False 
        try:
            if self.resume: 
                loaded = torch.load(trainsetFilename, map_location=torch.device(device))
                inputs, targets= loaded['inputs'], loaded['targets'], loaded['trivialpred']
                
                print("\ntrainset was loaded from checkpoint")
                resume_status = True  
            else:
                raise Exception()
        except:
            print("\nCreating new dataset..") 
            inputs = torch.randn((P,self.N))
            a_input = torch.sum(self.teacher_vec * inputs, dim=1)
            targets = a_input + a_input**2

        test_inputs = torch.randn((P_test,self.N))
        a_test = torch.sum(self.teacher_vec * test_inputs, dim=1)
        test_targets = a_test + a_test**2

        if conv: 
            inputs, test_inputs = inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3))), test_inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3)))


        return inputs, targets, test_inputs, test_targets, resume_status

    def training_type(self):
        return "train_synthetic", "test_synthetic"








class one_hl_dataset: 

    #@profile
    def __init__(self, teacher_vec, teacher_vec_2):
        self.teacher_vec = teacher_vec
        self.teacher_vec_2 = teacher_vec_2
        self.P = float("NaN")
        self.P_test = float("NaN")
        
        self.RGB = bool("NaN")
        self.N = len(self.teacher_vec[1])


    
    #@profile
    def make_data(self,P,P_test, conv, trainsetFilename, device ):
        resume_status = False 
        try:
            if self.resume: 
                loaded = torch.load(trainsetFilename, map_location=torch.device(device))
                inputs, targets, self.teacher_vec, self.teacher_vec_2 = loaded['inputs'], loaded['targets'], loaded['trivialpred'], loaded['teacher'], loaded['teacher2']
                
                print("\ntrainset was loaded from checkpoint")
                resume_status = True  
            else:
                raise Exception()
        except:
            print("\nCreating new dataset..") 
    
            inputs = torch.randn((P,self.N))
            targets = torch.tensordot(self.teacher_vec_2,nn.functional.relu(torch.tensordot(inputs, self.teacher_vec, dims=([-1], [-1]))),dims=([-1],[-1]))
        
        test_inputs = torch.randn((P_test,self.N))
        test_targets = torch.tensordot(self.teacher_vec_2,nn.functional.relu(torch.tensordot(test_inputs, self.teacher_vec, dims=([-1], [-1]))),dims=([-1],[-1]))
        
        if conv: 
            inputs, test_inputs = inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3))), test_inputs.view(-1,3, int(np.sqrt(self.N/3)), int(np.sqrt(self.N/3)))


        return  inputs, targets,  test_inputs, test_targets, resume_status

    def training_type(self):
        return "train_synthetic", "test_synthetic"
    
    def trivial_predictor(self, P_norm):

        inputs = torch.randn((P_norm,self.N))
        targets_pred = torch.tensordot(self.teacher_vec_2,nn.functional.relu(torch.tensordot(inputs, self.teacher_vec, dims=([-1], [-1]))),dims=([-1],[-1]))
        R0 = 0
        for t in targets_pred:
                R0 += t**2
        return (R0/P_norm).item()




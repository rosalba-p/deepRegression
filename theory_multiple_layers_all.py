from __future__ import print_function
import os, os.path, sys, time, math
import numpy as np


def k0_erf(x,y,lambda0,sigmab):
    N0 = len(x)
    Lambda = (1/(lambda0*N0))
    k0 = Lambda * np.dot(x,y)
    return k0 + sigmab**2

def kernel_erf(x,y,lambda1,lambda0,sigmab):
    k0xx = k0_erf(x,x,lambda0,sigmab)
    k0xy = k0_erf(x,y,lambda0,sigmab)
    k0yy = k0_erf(y,y,lambda0,sigmab)
    return (2/(lambda1*np.pi))*np.arcsin((2*k0xy)/np.sqrt((1+2*k0xx)*(1+2*k0yy)))+ sigmab**2



def correlation(x, y,lambda0):
    return np.dot(x,y)/(lambda0*len(x))

def normalized_dot(x,y):
    return correlation(x,y,1)/(np.sqrt(correlation(x,x,1)*correlation(y,y,1)))

def normalized_dot_sign(x,y):
    return np.dot(x,y)/(np.dot(x,x)*np.dot(y,y))

def kappa1(u):
    return (1/(2*np.pi))*(u * (np.pi - np.arccos(u))+ np.sqrt(1-u**2))

def kernel_sign(x,y):
    if (x == y).all():
        return 1
    else:
        #temp = correlation(x,y)/(np.sqrt(correlation(x,x)*correlation(y,y)))
        return (2/np.pi)*(np.arcsin(normalized_dot_sign(x,y)))

def kernel_ReLU(x,y,lambda0): 
    corrx = correlation(x,x,lambda0)
    corry = correlation(y,y,lambda0)

    f = (x == y)

    if f.all():
        return np.sqrt(corrx*corry)*kappa1(1)
    else:
        u = normalized_dot(x,y)
        kappa = kappa1(u)
        #print(u,kappa)
    return np.sqrt(corrx*corry)*kappa

def kernel_ReLU_2hl(x,y):
    corrx = kernel_ReLU(x,x)
    corry = kernel_ReLU(y,y)

    f = (x == y)
    if f.all():
        return np.sqrt(corrx*corry)
    else:
        u = normalized_dot(x,y)
        kappa = kappa1(u)
        #print(u,kappa)
        return np.sqrt(corrx*corry)*kappa

def kmatrix(data, lambda0):
    P = len(data)
    K = np.random.randn(P,P)
    for i in range(P): 
        for j in range(P):         
            K[i][j] = kernel_ReLU(data[i], data[j], lambda0)
    return K

def gen_error_1hl(data, x,y,labels, lambda1, invK, Qbar,lambda0):
    P = len(data)
    sum1 = 0
    sum2 = 0
    K0 = np.array([kernel_ReLU(x, data[mu],lambda0) for mu in range(P)])
    K0_invK = np.matmul(K0, invK)
    sum1 = -np.dot(K0_invK, labels) + y
    sum2 = -np.dot(K0_invK, K0) + kernel_ReLU(x,x, lambda0)
    #Qbar = qbar(labels, invK, N1)
    return sum1**2 - (Qbar)*sum2/lambda1


def gen_error_1hl_sign(data, x,y,labels, lambda1, invK, Qbar,lambda0):
    P = len(data)
    sum1 = 0
    sum2 = 0
    K0 = np.array([kernel_erf(x, data[mu],lambda1,lambda0,0.) for mu in range(P)])
    K0_invK = np.matmul(K0, invK)
    sum1 = -np.dot(K0_invK, labels) + y
    sum2 = -np.dot(K0_invK, K0) + kernel_erf(x,x,lambda1,lambda0,0.)
    #Qbar = qbar(labels, invK, N1)
    return sum1**2 - (Qbar)*sum2/lambda1

def qbar(labels, invK, N1,lambda1):
    P = len(labels)
    alpha1 = P/N1
    yky = np.matmul(np.matmul(np.transpose(labels), invK), labels)
    print(f'y K-1 y is {yky/P}')
    return ((alpha1-1)-np.sqrt((alpha1-1)**2 + lambda1*4*alpha1*yky/(P)))/2

def compute_theory_synthetic(inputs, targets, test_inputs, test_targets, N1, lambda1, first_subdir,P,Ptest,lambda0):
    data2 = np.array(inputs.flatten(start_dim = 1).cpu().clone().detach())
    N0 = len(data2[0])
    targets = targets.cpu().numpy()
    K = kmatrix(data2,lambda0)
    invK = np.linalg.inv(K)
    f = open(f"{first_subdir}theory_N_{N0}_N1_{N1}_lambda0_{lambda0}_lambda1_{lambda1}.txt", "a")
    Qbar = qbar(targets, invK, N1, lambda1)
    #Qbar = -1.
    #print(test_inputs[0])
    #x = np.array(test_inputs[0].clone().detach())
    #print(type(x))
    #y = np.array(test_targets[0].squeeze(0))
    #print(type(y))
    gen_error_pred = 0
    for p in range(Ptest):
        x = np.array(test_inputs[p].cpu().clone().detach())
        #print(type(x))
        y = np.array(test_targets[p].cpu().clone().detach().squeeze(0))
        #print(type(y))
        gen_error_pred += gen_error_1hl(data2, x, y, targets, lambda1, invK, Qbar,lambda0).item()
    #gen_error_pred = gen_error_1hl(data2, x, y, targets, lambda1, invK, Qbar).item()
    gen_error_pred = gen_error_pred/Ptest
    print(P, gen_error_pred, Qbar, file = f)
    print("\nPredicted error is: ", gen_error_pred, "\n\nStarting training")
    f.close
    return gen_error_pred, Qbar

def compute_theory_synthetic_sign(inputs, targets, test_inputs, test_targets, N1, lambda1, first_subdir,P,Ptest,lambda0):
    data2 = np.array(inputs.flatten(start_dim = 1).cpu().clone().detach())
    N0 = len(data2[0])
    targets = targets.cpu().numpy()
    K = kmatrix(data2,lambda0)
    invK = np.linalg.inv(K)
    f = open(f"{first_subdir}theory_N_{N0}_N1_{N1}_lambda0_{lambda0}_lambda1_{lambda1}.txt", "a")
    Qbar = qbar(targets, invK, N1, lambda1)
    #Qbar = -1.
    #print(test_inputs[0])
    #x = np.array(test_inputs[0].clone().detach())
    #print(type(x))
    #y = np.array(test_targets[0].squeeze(0))
    #print(type(y))
    gen_error_pred = 0
    for p in range(Ptest):
        x = np.array(test_inputs[p].cpu().clone().detach())
        #print(type(x))
        y = np.array(test_targets[p].cpu().clone().detach().squeeze(0))
        #print(type(y))
        gen_error_pred += gen_error_1hl_sign(data2, x, y, targets, lambda1, invK, Qbar,lambda0).item()
    #gen_error_pred = gen_error_1hl(data2, x, y, targets, lambda1, invK, Qbar).item()
    gen_error_pred = gen_error_pred/Ptest
    print(P, gen_error_pred, Qbar, file = f)
    print("\nPredicted error is: ", gen_error_pred, "\n\nStarting training")
    f.close
    return gen_error_pred, Qbar

def compute_theory(data, labels, testset_small, N1, lambda1, first_subdir,P):
    data2 = data.flatten(start_dim = 1).detach().numpy()
    labels2 = labels.detach().numpy()/10
    K = kmatrix(data2)
    invK = np.linalg.inv(K)


    f = open(f"{first_subdir}theory_ntk_mnist_N1_{N1}.txt", "a")
    #Qbar = -1
    Qbar = qbar(labels2, invK, N1, lambda1)
    x = np.array(testset_small[0][0].flatten(start_dim = 1))
    #x = np.array(testset_small[0][0].flatten(start_dim = 1).squeeze(0))

    y = np.array(testset_small[0][1]/10)

    gen_error_pred = gen_error_1hl(data2, x,y,labels2, lambda1, invK, Qbar)
    print("\nPredicted error is: ", gen_error_pred, "\n\nStarting training")

    print(P,gen_error_pred, file = f)
    f.close
    return gen_error_pred, Qbar







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:57:17 2020

@author: raphael-attias
"""

import torch
import pdb
import numpy as np
from mpmath import *
import matplotlib as plt
import matplotlib.patches as mpatches
import pylab as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from random import sample 
from scipy.stats import truncnorm
from scipy.integrate import quadrature
import time
import itertools


pl.clf()

def truef(x,fun): # Define the true function for our label y_data
    # Define Set of Functions
    a = 3
    b = 1/4
    def f0(x):
        n = 10
        y = 0
        for i in range(n):
            y = y + (1**n)*np.cos(np.pi*x*1.9**n)
        return y
    f1 = lambda t : np.cos(2*pl.pi*b+a*t)
    f2 = lambda t : (a**(-2)+(t-b)**2)**(-1)
    f3 = lambda t : (1+a*t)
    f4 = lambda t : np.exp(-a**2 * (t-b)**2)
    f5 = lambda t : np.exp(-a*np.abs(t-b))
    f6 = lambda t : 1/(1+25*x**2)
    f8 = lambda t : np.sin(4*pl.pi*x)*np.exp(-np.abs(5*x))
    def f9(x):
        return np.interp(x,np.linspace(-1,1,6),[0,1,0.4,1,0.8,0])
    
    def f7(x):  
        return 1/(x**2+1)
    F = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9]

    return F[fun](x)

def MLEf(x,x_data,y_data):
    
    X = x_data.numpy()
    Y = y_data.numpy()
    X = np.array([np.concatenate((np.array([1]),X[i,:])) for i in range(X.shape[0])])
    B = pl.inv(X.T@X)@X.T@Y
    s = 0
    N = X.shape[1]
    for i in np.arange(N):
        s = s + B[i]*(x**(i))
    return s,B.reshape(1,-1)

def legendrePoints(N):
    lp =  mp.polyroots(mp.taylor(lambda x: mp.legendre(N, x), 0, N)[::-1])
    s = np.array([float(lp[i]) for i in np.arange(len(lp))])
    return s

def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)

def chebychevPoints(N):
    arr = []
    for k in range(1,N):
        arr.append(np.cos((2*k-1)*np.pi/(2*N)))
    return arr

def datasample(N,fun,a,b,legendre):
    type = 1
    if type == 1:
        train_inputs = truncnorm.rvs(a,b,size=N)
        val_inputs = truncnorm.rvs(a,b,size=int(N*3))
    elif type == 2:
        train_inputs = np.random.uniform(a,b,N)
        val_inputs = np.random.uniform(a,b,int(N*3))
        
    train_labels = truef(train_inputs, fun)
    val_labels = truef(val_inputs,fun)
        
    return train_inputs.reshape(-1,1).tolist(),train_labels.reshape(-1,1).tolist(),val_inputs.reshape(-1,1).tolist(),val_labels.reshape(-1,1).tolist()

class datagen(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,N,deg,fun,a,b,legendre):
        self.train_inputs,self.train_labels,self.val_inputs,self.val_labels = datasample(N,fun,a,b,legendre)
        self.train_inputs, self.train_labels,self.val_inputs,self.val_labels = torch.tensor(self.train_inputs),torch.tensor(self.train_labels),torch.tensor(self.val_inputs),torch.tensor(self.val_labels) 
        self.len = N

    def __getitem__(self, index):
        return  self.train_inputs[index],self.train_labels[index]

    def __len__(self):
        return self.len
    
    def get_val(self):
        return self.val_inputs,self.val_labels
    
    def get_train(self):
        return self.train_inputs, self.train_labels
    
def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))    

def maclaurin(x,n):
    return np.sum(np.array([(-1)**i * x**(2*i) for i in range(0,n+1)]),axis=0)

def weierstrass(x, N):
	y = np.zeros((1,M))
	for n in xrange(1,N):
		y = y + np.cos(3**n*np.pi*x)/2**n
	return y


def Interpol(N,neurons,iter,initial_weights,initial_bias,fun=0,a=1,b=1):

    datasamp = datagen(N,neurons,fun,a,b,legendre)
    val_inputs,val_labels = datasamp.get_val()
    train_inputs,train_labels = datasamp.get_train()
    train_loader = DataLoader(dataset=datasamp,num_workers=0)# Initiate the data and labels
    
    class LockedCybenko(torch.nn.Module): # Cybenko with inner weight=1 and bias=-x[i]
        def __init__(self):
            super(LockedCybenko,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons,bias=True)
            self.fc1.weight.data = torch.ones(neurons).reshape(-1,1)
            self.fc1.bias.data = -torch.linspace(-1,1,neurons).reshape(1,-1).float()
            self.fc1.weight.requires_grad_(False)
            self.fc1.bias.requires_grad_(False)
            self.fc2 = torch.nn.Linear(neurons,1,bias=False)
            self.fc2.weight.data = initial_weights.reshape(1,-1).float()
            self.relu = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
        
    class SemilockedCybenko(torch.nn.Module): # Cybenko with inner weight=-1, and free bias
        def __init__(self):
            super(SemilockedCybenko,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons,bias=True)
            self.fc1.weight.data = torch.ones(neurons).reshape(-1,1).float()
            self.fc1.weight.requires_grad_(False)
            self.fc1.bias.requires_grad_(True)
            self.fc1.bias.data = initial_bias.reshape(1,-1).float()
            
            self.fc2 = torch.nn.Linear(neurons,1,bias=False)
            self.fc2.weight.data = initial_weights.reshape(1,-1).float()
            self.relu = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
        
    class UnlockedCybenko(torch.nn.Module): # Cybenko with free inner weight or bias
        def __init__(self):
            super(UnlockedCybenko,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons,bias=True)
            self.fc2 = torch.nn.Linear(neurons,1,bias=False)
            self.relu = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    class Network(torch.nn.Module): # Arbitrary network
        def __init__(self):
            super(Network,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons,bias=True)
            self.fc2 = torch.nn.Linear(neurons,2*neurons,bias=True)
            self.fc3 = torch.nn.Linear(2*neurons,1,bias=True)
            self.relu = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SemilockedCybenko()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.3)
    
    EL2Val = []
    EL2train = []
    ELinf = []
    EL2 = [] # L2 integral between f and u_teta
    
    optimizer = torch.optim.SGD(model.parameters(),lr=0.08)
    
    for epoch in range(iter):

        for i, (inputs,labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        def modelonx(x):
            return model(torch.tensor(x.reshape(-1,1).tolist(),requires_grad=False)).data.numpy().reshape(1,-1)
    
        def L2error(x):
            return (modelonx(x)-np.array(truef(x,fun)).reshape(1,-1))**2
        
        EL2Val.append(criterion(val_labels,model(val_inputs)))
        EL2train.append((criterion(train_labels,model(train_inputs))))
       # print(f'Epoch: {epoch} L2 Error on training : {EL2train[-1]:.6e} | L2 Error on validation : {EL2Val[-1]:.6e} | L2 on [-1,1] : {EL2[-1]:.6e}')
        if epoch == iter:   
            
            fig, ax = pl.subplots(nrows=1, ncols=2)
            plotrange = np.linspace(a-0.1,b+0.1,100)
            
            """ Function and Model Plot"""
            ax[0].scatter(val_inputs.data.numpy(),val_labels.data.numpy(),c='red',s=15)
            ax[0].scatter(train_inputs,train_labels,s=15)
            ax[0].plot(plotrange,model(torch.linspace(a-0.1,b+0.1,100).reshape(-1,1)).data.numpy(),'r')
            
            """ # Code qui permet d'afficher la fonction linéaire par morceau
            alpha = model.fc2.weight.data.numpy()[0]
            X = -model.fc1.bias.data.numpy()[0]
            ReLU = lambda t : np.where(t<=0,0,t)
            ax[0].plot(xx,alpha[0]*ReLU(xx-X[0])+alpha[1]*ReLU(xx-X[1])+alpha[2]*ReLU(xx-X[2])+alpha[3]*ReLU(xx-X[3])+alpha[4]*ReLU(xx-X[4])+alpha[5]*ReLU(xx-X[5]))
            """

            ax[0].plot(plotrange,truef(plotrange,fun),c='blue') 
            #ax[0].plot(np.linspace(a-0.1,b+0.1,100),np.polyval(np.polyfit(train_inputs.data.numpy().reshape(1,-1)[0],train_labels.data.numpy().reshape(1,-1)[0],10),np.linspace(a-0.1,b+0.1,100)),c='green')
            if fun == 7:
                ax[0].plot(plotrange,maclaurin(plotrange,50),c='green')
                ax[0].set_ylim(-0.1,1.1)

            """ Error Plot """
            ax[1].semilogy(range(epoch+1),EL2Val,color='red')
            ax[1].semilogy(range(epoch+1),EL2train,color='blue')
            #ax[1].semilogy(range(epoch+1),EL2,color='magenta')
            #ax[1].semilogy(range(epoch+1),ELinf,color='black')
            pl.show()   
    return EL2train[-1].detach().numpy()

np.random.seed(0)
torch.manual_seed(0)

def ListInitWeights(Range,Nb):
    return np.array([np.array(i) for i in itertools.product(Range, repeat=Nb)])

batch_size = 8;
batch_id = 1

totalWeightList = ListInitWeights(np.linspace(-5,5,3),6)
totalBiasList = ListInitWeights(np.linspace(-2,2,2),6)
totalLossList = np.zeros((len(totalBiasList),len(totalWeightList)))
totallen = len(totalWeightList)*len(totalBiasList[(batch_id-1)*batch_size:(batch_id)*batch_size])


k = 0
for i in range((batch_id-1)*batch_size,(batch_id)*batch_size):
    for j in range(len(totalWeightList)):
        t = time.time()
        np.random.seed(0)
        k =k+1
        totalLossList[i,j] = Interpol(30,6,50,torch.tensor(totalWeightList[j,:]),torch.tensor(totalBiasList[i,:]),9,-1,1)
        print(f'Weight {k}/{totallen}, Loss {totalLossList[i,j]}, Weight & Bias {totalWeightList[j,:]}, {totalBiasList[i,:]}, Time : {time.time()-t}, Time remaining : {(totallen-k)*(time.time()-t)/360:.2f} h')
  
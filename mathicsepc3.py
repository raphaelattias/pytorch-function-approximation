"""
Created on Sat Apr 11 09:57:17 2020

@author: raphael-attias
"""

import torch
import pdb
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from random import sample 
from scipy.stats import truncnorm
from scipy.integrate import quadrature
import time
import itertools


def truef(x): # Define the true function for our label y_data
    
    class target(torch.nn.Module): # Arbitrary network
        def __init__(self):
            super(target,self).__init__()
            self.fc1 = torch.nn.Linear(1,2,bias=True)
            self.fc2 = torch.nn.Linear(2,1,bias=False)
            self.relu = torch.nn.ReLU()
            self.fc1.bias.data = torch.tensor([-0.0146,  0.9464]).reshape(1,-1).float()
            self.fc1.weight.data = torch.tensor([-0.4576,  0.2975]).reshape(-1,1).float()
            self.fc2.weight.data = torch.tensor([-0.9804,  -0.8320]).reshape(1,-1).float()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
        
    model = target()
    

    return model(Variable(torch.from_numpy(x)).reshape(-1,1).float()).detach().numpy()


def datasample(N,a,b):
    type = 1
    if type == 1:
        train_inputs = truncnorm.rvs(a,b,size=N)
        val_inputs = truncnorm.rvs(a,b,size=int(N*3))
    elif type == 2:
        train_inputs = np.random.uniform(a,b,N)
        val_inputs = np.random.uniform(a,b,int(N*3))
        
    train_labels = truef(train_inputs)
    val_labels = truef(val_inputs)
        
    return train_inputs.reshape(-1,1).tolist(),train_labels.reshape(-1,1).tolist(),val_inputs.reshape(-1,1).tolist(),val_labels.reshape(-1,1).tolist()

class datagen(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,N,deg,a,b):
        self.train_inputs,self.train_labels,self.val_inputs,self.val_labels = datasample(N,a,b)
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
    


def Interpol(N,neurons,iter,initial_bias,initialinnerweights,initialouterweights,a=1,b=1):

    datasamp = datagen(N,neurons,a,b)
    val_inputs,val_labels = datasamp.get_val()
    train_inputs,train_labels = datasamp.get_train()
    train_loader = DataLoader(dataset=datasamp,num_workers=0)# Initiate the data and labels
    
        
    class SemilockedCybenko(torch.nn.Module): # Cybenko with inner weight=-1, and free bias
        def __init__(self):
            super(SemilockedCybenko,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons,bias=True)
            self.fc1.weight.data = initialinnerweights.reshape(-1,1).float()
            self.fc1.weight.requires_grad_(False)
            self.fc1.bias.requires_grad_(True)
            self.fc1.bias.data = initial_bias.reshape(1,-1).float()
            self.fc2 = torch.nn.Linear(neurons,1,bias=False)
            self.fc2.weight.data = initialouterweights.reshape(1,-1).float()
            self.relu = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SemilockedCybenko()
    criterion = torch.nn.MSELoss(reduction="sum")
    
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
            
            EL2train.append((criterion(train_labels,model(train_inputs))))
    return EL2train[-1].detach().numpy()

np.random.seed(0)
torch.manual_seed(0)

def ListInitWeights(Range,Nb):
    return np.array([np.array(i) for i in itertools.product(Range, repeat=Nb)])

batch_size = 9;
batch_id = 3;


totalWeightList = ListInitWeights(np.linspace(-1,1,6),2)
totalBiasList = ListInitWeights(np.linspace(-1,1,6),2)
totalLossList3 = np.zeros((len(totalBiasList),len(totalWeightList),len(totalWeightList)))
totallen = len(totalWeightList)*len(totalWeightList)*((batch_id)*batch_size-(batch_id-1)*batch_size)




l = 0
for i in range((batch_id-1)*batch_size,(batch_id)*batch_size):
    for j in range(len(totalWeightList)):
        for k in range(len(totalWeightList)):
            l = l+1
            t = time.time()
            totalLossList3[i,j,k] = Interpol(30,2,100,torch.tensor(totalBiasList[i,:]),torch.tensor(totalWeightList[j,:]),torch.tensor(totalWeightList[k,:]),-1,1)
            print(f'Iteration: {l}/{totallen}, Machine: {batch_id}, Loss {totalLossList3[i,j,k]},Time : {time.time()-t}, Time remaining : {(totallen-l)*(time.time()-t)/3600:.2f} h ')

from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save("bf3.npy",totalLossList3)
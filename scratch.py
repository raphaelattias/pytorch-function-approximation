#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:03:09 2020

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
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from random import sample 
from scipy.stats import truncnorm
from scipy.integrate import quadrature
from torch_lr_finder import LRFinder
import matplotlib.animation as animation
from torch.autograd import Variable
from mpl_toolkits import mplot3d


class Network(torch.nn.Module): # Arbitrary network
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = torch.nn.Linear(1,neurons,bias=True)
        self.fc2 = torch.nn.Linear(neurons,1,bias=False)
        self.relu = torch.nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
neurons =2;

model = Network()

PATH = "/Users/raphael-attias/Library/Mobile Documents/com~apple~CloudDocs/Cours/Semestre 6 - 2019:2020/Projet/Code/BruteForce2.pth"
a = -1; 
b= 1;
plotrange = np.linspace(a-0.1,b+0.1,100)
pl.plot(plotrange,model(torch.linspace(a-0.1,b+0.1,100).reshape(-1,1)).data.numpy(),'r')
summary(model,(100,1))
torch.save(model.state_dict(),PATH)
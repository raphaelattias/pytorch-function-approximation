import torch
import pdb
import numpy as np
from mpmath import *
import matplotlib as plt
import matplotlib.patches as mpatches
import pylab as pl
import torch.nn.functional as F
from torchsummary import summary
from random import sample 
from scipy.stats import truncnorm

pl.clf()

def truef(x,fun,a,b): # Define the true function for our label y_data
    # Define Set of Functions
    f0 = lambda t : np.random.randn()**2
    f1 = lambda t : np.cos(2*pl.pi*b+a*t)
    f2 = lambda t : (a**(-2)+(t-b)**2)**(-1)
    f3 = lambda t : (1+a*t)
    f4 = lambda t : np.exp(-a**2 * (t-b)**2)
    f5 = lambda t : np.exp(-a*np.abs(t-b))
    f6 = lambda t : 1/(1+25*x**2)
    F = [f0,f1,f2,f3,f4,f5,f6]
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

def datasample(N,deg,fun,a,b,legendre):
    train_inputs = []
    train_labels = []
    val_inputs = []
    val_labels = []
    xrange_extended = []
    deg = 1
    type = 2
    if type == 1:
        xrange = truncnorm.rvs(-1,1,size=1000)
        val = truncnorm.rvs(-1,1,size=int(1000/5))
    elif type == 2:
        xrange = np.random.uniform(-1,1,1000)
        val = np.random.uniform(-1,1,int(1000/5))

    for x in np.sort(sample(xrange.tolist(),N)): # Create the sample inputs and their labels
       # y = np.random.randn()
        y = truef(x,fun,a,b)
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        
        train_inputs.append(s)
        train_labels.append([y])
    
    for x in np.sort(sample(val.tolist(),int(N))):
        y = truef(x,fun,a,b)
        s = []
        for i in np.arange(deg):
            s.append(x)
        
        val_inputs.append(s)
        val_labels.append([y])
        
    
    for x in np.linspace(-1,1,1000):
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        xrange_extended.append(s)
    
    xxrange= []
    for x in np.linspace(-1,1,1000):
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        xxrange.append(s)
        
    return train_inputs,train_labels,xrange_extended,xxrange,val_inputs,val_labels

def modelonxx(model):
    
    YY = []
    for i in np.arange(1):   
        h = 1/10
        xx = np.linspace(-1+i*h,-1+(i+1)*h,100)
        xx = torch.tensor(xx,requires_grad=False).view(1,-1).float()
        yy = model(xx)
        YY.append(yy.data.numpy())
        pl.plot(xx.data.numpy()[0],yy.data.numpy()[0])
        pl.show()
        input("waiting")
        
    return np.array(YY).reshape(1000,-1)
    
    

def Interpol(N,deg,iter,fun=0,a=1,b=1,displayReal=0,legendre=0):
    
    inputs, labels,xrange_extended,xrange,val_inputs,val_labels = datasample(N,deg,fun,a,b,legendre) # Initiate the data and labels
    
    class Net2(torch.nn.Module): # Initiate the network
        def __init__(self):
            super(Net2,self).__init__()
            self.fc1 = torch.nn.Linear(N,2*N)
            self.fc2 = torch.nn.Linear(2*N,int(N))
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self,x):
            x = self.sigmoid(self.fc1(x))
            return self.fc2(x)
            
    model = Net2()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    
    y_data = torch.tensor(labels,requires_grad=False)
    x_data = torch.tensor(inputs,requires_grad=False).view(1,-1)
    val_labels = torch.tensor(val_labels,requires_grad=False).view(1,-1)
    val_inputs = torch.tensor(val_inputs,requires_grad=False).view(1,-1)
    
    xx = np.linspace(-1,1, 1000)
    #pl.plot(xx,truef(xx,fun,a,b))
    
    pl.show()
    EL2Val = []
    EL2train = []
    ELinf = []
    #input("Press Enter to continue...")
    for epoch in range(iter):
        y_pred = model(x_data)
        loss = criterion(y_pred,y_data.view(1,-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        EL2train.append(loss)
        EL2Val.append(np.sum((val_labels.data.numpy()-model(val_inputs).data.numpy())**2)/val_labels.data.numpy().shape[1])
       # modelonxx(model)
        pl.plot(x_data.data.numpy()[0],y_pred.data.numpy()[0],c='r')
        pl.show()
    
    pl.scatter(x_data,y_data,c='b')
    pl.scatter(val_inputs,val_labels,c='r')
    pl.show()
    pl.plot(range(iter),EL2Val,color='blue')
    pl.plot(range(iter),EL2train,color='g')
    plt.pyplot.xlabel('$n$')
    plt.pyplot.ylabel('$e_n$')
    plt.pyplot.yscale("log")
    pl.show()
    summary(model, input_size=(1, N))
    
    return EL2Val
EL2Val = Interpol(100,3,50,6,3,1/2,1,0)
# Interpol(N,deg,epoch,fun=2, a=1, b=2, displayReal=1,typePoint=0)
# N : number of regression points
# deg : degree of interpolating polynomial
# epoch : number of iterations of backward steepest descent
# fun : function to interpolate
#       fun = 0 : random point with  y = normal distribution
#       fun = 1 : GENZ1 Function : Oscillatory cos(2*pi*b+a*x)
#       fun = 2 : GENZ2 Function : Product Peak
#       fun = 3 : GENZ3 Function : Corner Peak 
#       fun = 4 : GENZ4 Function : Gaussian 
#       fun = 5 : GENZ5 Function : C0 function
#       fun = 6 : Runge Function : 1/(1+25*x**2) 
# a : first parameter needed for fun
# b : second parameter needed for fun
# displayReal : BOOL that displays the real function if set to TRUE
# typePoints : 0 for equidistant, 1 for Legendre, 2 for Chebychev

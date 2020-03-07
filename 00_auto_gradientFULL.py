import torch
import pdb
import numpy as np
from mpmath import *
import matplotlib as plt
import matplotlib.patches as mpatches
import pylab as pl
import torch.nn.functional as F
from torchsummary import summary


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
    inputs = []
    labels = []
    xrange_extended = []
    if legendre == 1:
        xrange = legendrePoints(N)
    elif legendre == 2:
        xrange = chebychevPoints(N)
    else:
        xrange = np.linspace(-1,1,N)
        
    for x in xrange: # Create the sample inputs and their labels
       # y = np.random.randn()
        y = truef(x,fun,a,b)
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        
        inputs.append(s)
        labels.append([y])
    
    for x in np.linspace(-1.1,1.1,1000):
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        xrange_extended.append(s)
    xrange = []
    for x in np.linspace(-1,1,1000):
        s = []
        for i in np.arange(deg):
            s.append(x**(i+1))
        xrange.append(s)
        
    return inputs,labels,xrange_extended,xrange

def combinatorics(n):
    arr = []
    arr2 = np.array([(k,n-k) for k in range(n+1)])
    arr.append(arr2)
    return arr

def Interpol(N,deg,iter,fun=0,a=1,b=1,displayReal=0,legendre=0):
    
    inputs, labels,xrange_extended,xrange = datasample(N,deg,fun,a,b,legendre) # Initiate the data and labels
    class Net1(torch.nn.Module): # Initiate the network
        def __init__(self):
            super(Net1,self).__init__()
            self.fc1 = torch.nn.Linear(deg,1)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self,x):
            x = self.fc1(x)
            return x
    
    class Net2(torch.nn.Module): # Initiate the network
        def __init__(self):
            super(Net2,self).__init__()
            self.fc1 = torch.nn.Linear(deg,5)
            self.fc2 = torch.nn.Linear(5,2)
            self.fc3 = torch.nn.Linear(2,1)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self,x):
            x = self.sigmoid(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return self.fc3(x)
            
    model = Net2()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    
    y_data = torch.tensor(labels,requires_grad=False)
    x_data = torch.tensor(inputs,requires_grad=False)
    xrange_extended = torch.tensor(xrange_extended,requires_grad=False)
    xrange = torch.tensor(xrange,requires_grad=False)
    ETetaTrue = []
    ETetaTetastar = []
    for epoch in range(iter):
        y_pred = model(x_data)
        loss = criterion(y_pred,y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        yrange_formated = model(xrange_extended).data.numpy()
        
        AbsError = np.abs(model(xrange).data.numpy().reshape(-1,1)-truef(np.linspace(-1,1,1000),fun,a,b).reshape(-1,1))
        MaxError = np.max(AbsError)
        MaxErrorIndex = round(np.linspace(-1,1,1000)[np.argmax(AbsError)],4)
        print(f'Epoch: {epoch} | Loss: {loss} | MaxError : {MaxError}, Index : {MaxErrorIndex}')
        ETetaTrue.append(MaxError)
        
       # AbsError2 = np.abs(model(xrange).data.numpy().reshape(-1,1)-np.array([MLEf(np.linspace(-1,1,1000),x_data,y_data)]).reshape(-1,1))
       # ETetaTetastar.append(np.max(AbsError2))
    def predf(x):
        s = 0
        N = len(model.fc1.weight.tolist()[0])
        for i in np.arange(N):
            s = s + model.fc1.weight.tolist()[0][i]*(x**(i+1))
            
        return s+model.fc1.bias.item()
    
    xx = np.linspace(-1.1,1.1,1000)
    if legendre == 1:
        xrange = legendrePoints(N)
    elif legendre == 2:
        xrange = chebychevPoints(N)
    else:
        xrange = np.linspace(-1,1,N)
        
    print("Model MLE :",  MLEf(xx, x_data, y_data)[1])
    print("Model Prediction (after training) :", model.fc1.weight.data.numpy())
    pl.scatter(xrange,y_data.data,c='blue')
    pl.plot(xx,yrange_formated,c='red') # Plot the predicted f by the model
   # pl.plot(xx,MLEf(xx, x_data, y_data)[0],c='green')
    green_patch = mpatches.Patch(color='green', label=r'$u^*_{\theta}$')
    red_patch = mpatches.Patch(color='red', label=r'$u_{\theta}$')
    if displayReal and fun != 0:
        pl.plot(xx,truef(xx,fun,a,b),c='blue')
        blue_patch = mpatches.Patch(color='blue', label=r'$f$')
        pl.legend(handles=[red_patch,blue_patch,green_patch])
    else:
        pl.legend(handles=[red_patch,green_patch])
    #pl.scatter(np.linspace(-1,1,N),model(x_data).data,c='red')
    plt.pyplot.ylim(top=1.5,bottom=-0.5)
    plt.pyplot.show()
    pl.plot(range(iter),ETetaTrue,color='r')
  #  pl.plot(range(iter),ETetaTetastar,color='g')
    plt.pyplot.xlabel('$n$')
    plt.pyplot.ylabel('$e_n$')
    plt.pyplot.yscale("log")
    plt.pyplot.show()


Interpol(50,49,1000,6,3,1/2,1,0)
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
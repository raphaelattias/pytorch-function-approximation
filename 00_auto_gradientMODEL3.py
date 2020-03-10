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
    f8 = lambda t : np.sin(4*pl.pi*x)*np.exp(-np.abs(5*x))
    
    def f7(x):  
        y = np.zeros(len(x))
        y[x < 0] = float(-5*x)
        y[x >= 0] = float(sin(x)**2)
        return y
        
    F = [f0,f1,f2,f3,f4,f5,f6,f7,f8]
    


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
        train_labels.append([y+np.random.randn()/30])
    
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
        
    return train_inputs,train_labels,val_inputs,val_labels

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
    
class datagen(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,N,deg,fun,a,b,legendre):
        self.train_inputs,self.train_labels,self.val_inputs,self.val_labels = datasample(N,deg,fun,a,b,legendre)
        self.train_inputs, self.train_labels,self.val_inputs,self.val_labels = torch.tensor(self.train_inputs),torch.tensor(self.train_labels),torch.tensor(self.val_inputs),torch.tensor(self.val_labels) 
        self.len = N

    def __getitem__(self, index):
        return  self.train_inputs[index],self.train_labels[index]

    def __len__(self):
        return self.len
    
    def get_val(self):
        return self.val_inputs,self.val_labels
    

def Interpol(N,neurons,iter,fun=0,a=1,b=1,displayReal=0,legendre=0):
    
    datasamp = datagen(N,neurons,fun,a,b,legendre)
    val_inputs,val_labels = datasamp.get_val()
    train_loader = DataLoader(dataset=datasamp,num_workers=0)# Initiate the data and labels
    print(len(train_loader))
    class Net3(torch.nn.Module): # Initiate the network
        def __init__(self):
            super(Net3,self).__init__()
            self.fc1 = torch.nn.Linear(1,neurons)
            self.fc2 = torch.nn.Linear(neurons,1)
            self.sigmoid = torch.nn.ReLU()
        
        def forward(self,x):
            x = self.sigmoid(self.fc1(x))
            return self.fc2(x)
            
    model = Net3()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.3)
    
    xx = np.linspace(-1,1, 1000)
    #pl.plot(xx,truef(xx,fun,a,b))
    
    pl.show()
    EL2Val = []
    EL2train = []
    ELinf = []
    #input("Press Enter to continue...")
    
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader, start_lr=0.001, end_lr=1.5, num_iter=1000)
    lr_finder.reset() # to reset the model and optimizer to their initial state
    learning = lr_finder.history.get('lr')[np.argmin(lr_finder.history.get('loss'))]
    
    optimizer = torch.optim.SGD(model.parameters(),lr=learning)
    
    for epoch in range(iter):
        x = []
        ytrue = []
        ypred = []
        for i, (inputs,labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            
            x.append(inputs.data.numpy())
            ytrue.append(labels.data.numpy())
            ypred.append(y_pred.data.numpy())     
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        EL2Val.append(criterion(val_labels,model(val_inputs)))
        EL2train.append((criterion(datasamp[:][1],model(datasamp[:][0]))))
        print(f'Epoch: {epoch}, LR : {learning}  L2 Error on training : {EL2train[-1]} | L2 Error on validation : {EL2Val[-1]} ')

       # modelonxx(model)
        if epoch % 10 == 0:   
            pl.scatter(x,ytrue,c='blue')
            pl.scatter(val_inputs.data.numpy(),val_labels.data.numpy(),c='red')
            pl.plot(np.array(x).reshape(1,-1)[0],np.array(ypred).reshape(1,-1)[0])
            pl.plot(np.linspace(-1,1,100),truef(np.linspace(-1,1,100),fun,a,b),c='blue')
            pl.show()
    pl.plot(range(iter),EL2Val,color='blue')
    pl.plot(range(iter),EL2train,color='g')
    plt.pyplot.xlabel('$n$')
    plt.pyplot.ylabel('$e_n$')
    plt.pyplot.yscale("log")
    pl.show()
    
    def modelf(x):
        s = []
        for x in np.arange(x.shape[0]):
            s.append(model(double(x)))
        return s
    
    return model
model = Interpol(100,100, 1000,1,3,1/2,1,0)
# Interpol(N,neurons,epoch,fun=2, a=1, b=2, displayReal=1,typePoint=0)
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

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.autograd import Variable


class online_update_layer(torch.nn.Module):
    def __init__(self, d, eta, x=None, dtype=torch.float64):
        super(online_update_layer, self).__init__()
        self.dim = d
        self.x = Parameter(torch.Tensor(d))
        self.eta = eta
        self.dtype = dtype
        # Initialize the weights
        if x is not None:
            self.x.data = x
        else:
            stdv = 1. / np.sqrt(d)
            self.x.data.uniform_(-stdv, stdv)
            
    def forward(self, theta):
        pred = torch.dot(self.x, theta)
        grad = -self.x/(1.+torch.exp(pred))
        theta_next = theta - self.eta*(grad)
        return theta_next
    
class online(nn.Module):

    def __init__(self, d, eta, n, x=None, dtype=torch.float64):

        super(online, self).__init__()
        
        self.d, self.eta, self.n = d, eta, n
        self.dtype = torch.float64
        # Hidden layers
        self.layers = nn.ModuleList()
        for k in range(n):
            self.layers.append(online_update_layer(d,eta,x))

    def forward(self, theta):

        # Feedforward
        for layer in self.layers:
            theta = layer(theta)
        
        return theta
    
    def get_xs(self):
        d, n = self.d, self.n
        xs = torch.zeros([n,d],dtype=self.dtype)
        for i,layer in enumerate(self.layers):
            xs[i,:] = layer.x
        return xs
    
    def set_xs(self, xs):
        d, n = self.d, self.n
        #for i, layer in enumerate(self.layers):
        for i in range(n):
            self.layers[i].x.data = xs[i,:]
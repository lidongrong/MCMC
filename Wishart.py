# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:03:21 2021

@author: lidongrong98
"""


import numpy as np
import torch
import math

# n>p parameters for wishart 
n=3
V=np.eye(2)
p=2

def multivariate_gamma(n,p):
    cum=1
    for i in range(0,p):
        cum=math.gamma(n-p/2)*cum
    return np.pi**(p*(p-1)/4)*cum


# important numbers for wishart
gamma_p=multivariate_gamma(n/2,p)
V_det=np.linalg.det(V)
V_det=V_det**(n/2)

x=torch.eye(2,requires_grad=True)

#Wishart distribution pdf
def Wishart(x,n,p,V):
    temp1=1/(2**(n*p/2)*V_det*gamma_p)*(torch.pow(torch.trace(x),(n-p-1)/2))
    V_new=torch.from_numpy(V)
    temp2=torch.exp(-0.5*
                    torch.trace(
                        torch.inverse(V_new)*x))
    return temp1*temp2



    

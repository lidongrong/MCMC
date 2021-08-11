# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:35:16 2021

@author: lidongrong98
"""


import torch
import numpy as np
import math


# p should be a function taking a tensor as  (requires_grad shall be set to True)
# theta0 is the initial value for iteration, should be a tensor
class Sampler():
    def __init__(self,p):
        #self.theta0=theta0
        #probability density function, shall be a torch.tensor object
        self.p=p
        # learning rate, 0.1 as default
        self.rate=0.1
        # number of iterations for each leapfrog, 15 as default
        self.Leapfrog_step=15
        
        
    #sample from target distribution using Hamiltonian Dynamic
    #draws: number of samples
    #eps: step size
    def HMC(self,draws,theta0):
        eps=self.rate
        samples=[]
        # reset theta to make sure that it is a leaf node in computation graph
        theta=theta0.tolist()
        theta=torch.tensor(theta,requires_grad=True)
        
        for i in range(0,draws):
            '''
            # evaluate log pdf
            logpdf=torch.log(self.p(theta))
            # compute the gradient
            logpdf.backward()
            # generate new value for each iteration
            new_theta=theta+eps*theta.grad+math.sqrt(2*eps)*torch.normal(0,1,theta.size())
            samples.append(new_theta.tolist())
            theta=torch.tensor(new_theta.tolist(),requires_grad=True)
            '''
            #print(i)
            # prepare to leapfrog
            phi=torch.normal(0,1,theta.size())
            #phi1=torch.tensor(phi.tolist())
            #theta1=torch.tensor(theta.tolist(),requires_grad=True)
            phi1=phi
            theta1=theta.clone().detach().requires_grad_(True)
            #Leapfrog
            for j in range(0,self.Leapfrog_step):
                logpdf=torch.log(self.p(theta1))
                logpdf.backward()
                phi1=phi1+0.5*eps*theta1.grad
                #theta1=torch.tensor((theta1+eps*phi1).tolist(),requires_grad=True)
                #theta1.data=(theta1+eps*phi1).data
                theta1=(theta1+eps*phi1).clone().detach().requires_grad_(True)
                logpdf=torch.log(self.p(theta1))
                logpdf.backward()
                phi1=(phi1+0.5*eps*theta1.grad)
            #Metropolis step
            r=(self.p(theta1)/self.p(theta))*torch.exp((torch.norm(phi)**2-
                                                        torch.norm(phi1)**2)/2)
            
            
            if torch.rand(1).item()<min(1,r.item()):
                #theta=torch.tensor(theta1.tolist(),requires_grad=True)
                theta=theta1.detach().clone().requires_grad_(True)
            samples.append(theta.tolist())
            #print(theta)
            #print(theta1)
        #output is a torch.tensor object
        samples=torch.tensor(samples)
        return samples
    


# Common densities
# parameters & inputs for density should be set to tensor!
# Normal distribution & multivariate normal
def normal(mean=torch.tensor(0.),std=torch.tensor(1.)):
    def norm(x):
        tmp1=1/((math.sqrt(2*math.pi)*std))
        tmp2=torch.exp(-0.5*(x-mean)*(x-mean)/std**2)
        return tmp1*tmp2
    return norm
                
# Multivariate normal distribution
def multi_normal(mean,std):
    def norm(x):
        tmp1=1/(((2*math.pi)**(len(mean)/2))*torch.sqrt(torch.det(std)))
        tmp2=torch.exp(-0.5*torch.matmul(x.T-mean.T,torch.matmul(torch.inverse(std),x-mean)))
        return tmp1*tmp2
    return norm

# Mixture of 2 dimensional Gaussian, just for testing
def mixture(x):
    mean1=torch.tensor([0.,0.])
    mean2=torch.tensor([3.,4.])
    std=torch.eye(2)
    p1=multi_normal(mean1,std)
    p2=multi_normal(mean2,std)
    return 0.5*p1(x)+0.5*p2(x)
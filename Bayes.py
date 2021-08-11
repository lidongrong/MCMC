# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:31:05 2021

@author: s1155151972
"""


import LMC


# High order function that returns function multiplication
def func_mul(p1,p2):
    def full(theta,data):
        return p1(theta)*p2(data)
    return full


# Bayesian Inference 
# data: data should be a tensor
# prior: prior density of coef theta
# likelihood: likelihood function of a single data point
# Full likelihood will be automatically computed by the code
class Model(LMC.Sampler):
    def __init__(self,prior,likelihood,data):
        self.prior=prior
        self.likelihood=likelihood
        self.data=data
        self.rate=0.1
        self.Leapfrog_step=15
        
        '''
        self.full=1
        for d in data:
            self.full=self.full*self.likelihood(d[:])
        
        def full(u):
            return self.full
        
        def full_dist(u):
            return self.prior(u)*full(u)
        
        self.p=full_dist
        '''
        def full(theta):
            full_pdf=self.prior(theta)
            for d in data:
                full_pdf=full_pdf*self.likelihood(d[:])
            return full_pdf
        self.p=full
        
        
        
        
                
        
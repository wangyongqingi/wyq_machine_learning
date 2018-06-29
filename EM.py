# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:44:58 2018

@author: wyq
"""

#高斯混合模型参数估计的EM算法  9.3


import numpy as np
from math import exp
import copy

class EM(object):
    def __init__(self,sigma,Mu1,Mu2,N,epsilon = 0.01,k = 2,IterateNumber = 20):
        self.sigma = sigma
        self.Mu1 = Mu1
        self.Mu2 = Mu2
        self.N = N
        self.epsilon = epsilon
        self.k = k
        self.IterateNumber = IterateNumber
        self.X = np.array([0.0 for x in range(N)])
        self.Mu = np.random.random(k)
        self.Expectations = np.zeros((N,k))
        for i in range(N):
            if np.random.random(1) > 0.5:
                self.X[i] = np.random.normal(Mu1,sigma)
            else:
                self.X[i] = np.random.normal(Mu2,sigma)
        
    def E_step(self):
        Expectations=self.Expectations
        sigma=self.sigma
        Mu=self.Mu
        X=self.X
        k = self.k
        N = self.N
        for i in range(N):
            fyin = [0.0 for x in range(k)]
            Fyin = 0.0
            for j in range(k):
                fyin[j] = exp(-(X[i]-Mu[j])**2/(2*sigma**2))
                Fyin += fyin[j]
            for j in range(k): 
                Expectations[i][j] = fyin[j] / Fyin
        self.Expectations = Expectations
        
    def M_step(self):
        Expectations = self.Expectations
        X = self.X
        k = self.k
        N = self.N
        for i in range(k):
            Elements = 0
            Sum = 0
            for j in range(N):
                Elements += Expectations[j][i]*X[j]
                Sum += Expectations[j][i]
            self.Mu[i] = Elements/Sum
        
    def stop_condition(self,Mu):
        condition = False
        if sum(abs(self.Mu - Mu)) < self.epsilon:
            condition = True
        return condition    
    
    def run(self):
        IterateNumber = self.IterateNumber
        for i in range(IterateNumber):
            OldMu = copy.deepcopy(self.Mu)
            self.E_step()
            self.M_step()
            print(i,self.Mu)
            if self.stop_condition(OldMu) == True:
                break

def main():
    sigma = 6 
    Mu1 = 40    
    Mu2 = 20    
    N = 1000  
    em=EM(sigma,Mu1,Mu2,N)
    em.run()

if __name__ == "__main__":
    main()     
                
            
        

     
        

    
        
        


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:48:30 2018

@author: wyq
"""

#照着书上的例子写了前向算法，后向算法以及维特比算法
import numpy as np

class HMM(object):
    def __init__(self,A,B,pi,O):
        self.A=A
        self.B=B
        self.pi=pi
        self.M=len(B[0])
        self.N=len(A)
        self.O=O
        self.T=len(O)
        self.alphas=np.zeros((self.T,self.N))
        self.beta=np.zeros((self.T,self.N))
        
    def Forward(self):
        N=self.N
        pi=self.pi
        A=self.A
        B=self.B
        P_O_lamda=0  #P(O|λ) 
        for i in range(N):
            self.alphas[0][i]=pi[i]*B[i][self.O[0]]
        for t in range(1,self.T):
            for i in range(N):
                Sum=0
                for j in range(N):
                    Sum+=self.alphas[t-1][j]*A[j][i]
                self.alphas[t][i]=Sum*self.B[i][self.O[t]] #公式10.16
        for i in range(self.T):
            P_O_lamda+=self.alphas[self.T-1][i]  #公式10.17
        return P_O_lamda,self.alphas
    
    def Backward(self):
        N=self.N
        pi=self.pi
        A=self.A
        B=self.B
        T=self.T
        O=self.O
        P_O_lamda=0 
        for i in range(N):
            self.beta[T-1][i]=1
        #公式10.20
        for t in range(T-2,-1,-1):
            for i in range(N):
                for j in range(N):
                    self.beta[t][i]+=A[i][j]*B[j][O[t+1]]*self.beta[t+1][j] 
        for i in range(N):
            P_O_lamda+=pi[i]*B[i][O[0]]*self.beta[0][i] #公式10.21
        return P_O_lamda,self.beta
    
    def Viterbi(self): #算法10.5,维特比算法
        N=self.N
        pi=self.pi
        A=self.A
        B=self.B
        T=self.T
        O=self.O
        self.theta=np.zeros((T,N))
        self.phi=np.zeros((T,N))
        I=np.zeros(T)
        for i in range(N):
            self.theta[0][i]=pi[i]*B[i][O[0]]
            self.phi[0][i]=0
        for t in range(1,T):
            for i in range(N):
                for j in range(N):
                    if self.theta[t][i]<=self.theta[t-1][j]*A[j][i]*B[i][O[t]]:
                        self.theta[t][i]=self.theta[t-1][j]*A[j][i]*B[i][O[t]]
                        self.phi[t][i]=j
        probablity=max(self.theta[T-1])
        for i in range(N):
            if self.theta[T-1][i]==probablity:
                I[T-1]=i
        for t in range(T-2,-1,-1):
            I[t]=self.phi[t+1][int(I[t+1])]
        return probablity,I        
           
def main():
    a=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]
    b=[[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    pi=[0.2,0.4,0.4]
    o=[0,1,0]
    hmm=HMM(a,b,pi,o)
    print(hmm.Forward())
    print(hmm.Backward())
    print(hmm.Viterbi())

if __name__=="__main__":
    main()


            
        
        
    
        
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:55:55 2018

@author: wyq
       
"""

import numpy as np

def linear_kernel(xi,xj):
    result=np.dot(np.array(xi),np.array(xj))
    return result

def gaussian_kernel(xi,xj):
    sigma=5
    x=np.array(xi)-np.array(xj)
    x_dot=np.dot(x,x)
    result=np.exp(-x_dot/(2*(sigma**2)))
    return result


class X_Y_Match_Error(TypeError):
    pass

class Iteration_ValueError(ValueError):
    pass


class SVM(object):
    def __init__(self,X,Y,C,maxIteration,epsilon=0.001,kernel='rbf'):
        self.X=X
        self.Y=Y
        self.C=500
        self.epsilon=epsilon
        self.maxIteration=maxIteration  #迭代次数
        self.kernel=kernel
        self.N=len(self.X)  
        self.n=len(self.X[0])
        self.b=0
        self.alpha=[1 for x in range(self.N)]
        self.E=[self.calErr(i) for i in range(self.N)]
        if(len(self.X)!=len(self.Y)):
            raise X_Y_Match_Error('the train data and label does not match')
        if(type(self.maxIteration)!=int):
            raise Iteration_ValueError('the type of n_iteration should be int')
        
    def kernelResult(self,xi,xj):
        if(self.kernel=='linear'):
            result=linear_kernel(xi,xj)
        elif(self.kernel=='rbf'):
            result=gaussian_kernel(xi,xj)
        return result
    
    def g_xi(self,i):
        result=self.b
        for j in range(self.N):
            result+=self.alpha[j]*self.Y[j]*self.kernelResult(self.X[i],self.X[j])
        return result
    
    def calErr(self,i):
        result=self.g_xi(i)-self.Y[i]
        return result

    def KKT_condition(self,i):#P130 7.4.3 SMO算法
        if ((self.Y[i]*self.E[i]<-self.epsilon) and (self.alpha[i]<self.C)) or \
        ((self.Y[i]*self.E[i]>self.epsilon) and (self.alpha[i]>0)):
            return False
        else:
            return True	
        
    def KKT_stop_condition(self):
        for i in range(self.N):
            satisfy_condition=self.KKT_condition(i)
            if satisfy_condition==False:
                return False
        return True
        
    def select_two_parameters(self):
        index_list=[i for i in range(self.N)]
        i1_list_1=[]
        for i in index_list:
            if(self.alpha[i]>0 and self.alpha[i]<self.C):
                i1_list_1.append(index_list[i])
        i1_list_2 = list(set(index_list) - set(i1_list_1))
        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self.KKT_condition(i):
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue
                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)

        return i, max_[1]
    
    def iteration_train(self):#P125 7.4.1 
        iterate=0
        while(iterate<self.maxIteration and self.KKT_stop_condition()==False):
            i, j = self.select_two_parameters()
            L = max(0, (self.alpha[j] - self.alpha[i]))
            H = min(self.C, (self.C + self.alpha[j] - self.alpha[i]))
            if self.Y[i] == self.Y[j]:
                L = max(0, (self.alpha[j] + self.alpha[i] - self.C))
                H = min(self.C, (self.alpha[j] + self.alpha[i]))
            E1 = self.E[i]
            E2 = self.E[j]
            eta=self.kernelResult(self.X[i], self.X[i])+self.kernelResult(self.X[j], self.X[j])-\
            2*self.kernelResult(self.X[i], self.X[j])
            alphaj_new_unc = self.alpha[j] + self.Y[j] * (E1 - E2) / eta      
            alphj_new = 0
            if alphaj_new_unc > H:
                alphj_new = H
            elif alphaj_new_unc < L:
                alphj_new = L
            else:
                alphj_new = alphaj_new_unc            
            alphi_new = self.alpha[i] + self.Y[i] * self.Y[j] * (self.alpha[j] - alphj_new)
            b_new = 0
            b1_new = -E1 - self.Y[i] * self.kernelResult(self.X[i], self.X[i]) * (alphi_new - self.alpha[i]) - self.Y[j] \
            * self.kernelResult(self.X[j], self.X[i]) * (alphj_new - self.alpha[j]) + self.b
            b2_new = -E2 - self.Y[i] * self.kernelResult(self.X[i], self.X[j]) * (alphi_new - self.alpha[i]) - self.Y[j] \
            * self.kernelResult(self.X[j], self.X[j]) * (alphj_new - self.alpha[j]) + self.b
            if alphi_new > 0 and alphi_new < self.C:
                b_new = b1_new
            elif alphj_new > 0 and alphj_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
                self.alpha[i] = alphi_new
                self.alpha[j] = alphj_new
                self.b = b_new
                self.E[i] = self.calErr(i)
                self.E[j] = self.calErr(j)
            iterate+=1

    def predict(self,feature):
        result = self.b
        for i in range(self.N):
            result += self.alpha[i]*self.Y[i]*self.kernelResult(feature,self.X[i])
        if result > 0:
            return 1
        else:
            return -1

def main():
    X=[[1,2,3],[2,3,4],[3,4,5],[4,5,6],[3,4,8],[5,7,7],[7,8,9]]
    Y=[1,1,-1,-1,-1,1,-1]
    svm=SVM(X,Y,C=10,maxIteration=100,epsilon=0.1,kernel='linear')
    svm.iteration_train()
    print(svm.predict([1,2,3]))
if __name__=='__main__':
    main()

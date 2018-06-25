# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 21:46:58 2018

@author: wyq
"""

#根据书上例子8.1写一个阈值分类器，二分类的标签为+1，-1, 也可采用其他基础分类器，将阈值分类改为其他基础分类器
from math import exp
from math import log
import numpy as np

class X_Y_train_Error(TypeError):
    pass

class Parameters(object):#定义一个计算各参数（阈值，误差率等）的类
    def __init__(self,X,Y,w,threshold):#其中w代表权重
        self.x=X
        self.y=Y
        self.w=w
        self.N=len(self.x)
        self.threshold=threshold
        self.order1=True
        if(len(self.x)!=len(self.y)):
            raise X_Y_train_Error('the train data X and predict label Y do not match' )
        
    def calc_order1(self):
        threshold=0
        error_rate=1
        val=[1 for x in range(self.N)]
        for i in self.threshold:
            err=0
            for j in range(self.N):
                if(self.x[j]<i):
                    val[j]=1
                else:
                    val[j]=-1
                if val[j]*self.y[j]<0:
                    err+=self.w[j]
            if err<error_rate:
                threshold=i
                error_rate=err
        return threshold,error_rate
    
    def calc_order2(self):
        threshold=0
        error_rate=1
        val=[1 for x in range(self.N)]
        for i in self.threshold:
            err=0
            for j in range(self.N):
                if(self.x[j]<i):
                    val[j]=-1
                else:
                    val[j]=1
                if val[j]*self.y[j]<0:
                    err+=self.w[j]
            if err<error_rate:
                threshold=i
                error_rate=err
        return threshold,error_rate
    
    def better_error(self):
        order1_threshold,order1_error_rate=self.calc_order1()
        order2_threshold,order2_error_rate=self.calc_order2()
        if order1_error_rate<=order2_error_rate:
            self.index=order1_threshold
            error_rate=order1_error_rate
        else:
            self.index=order2_threshold
            error_rate=order2_error_rate
            self.order1=False
        return error_rate
    
    def Gxi(self,xi):
        if self.order1==True:
            if xi<=self.index:
                return 1.0
            else:
                return -1.0
        else:
            if xi<=self.index:
                return -1.0
            else:
                return 1.0
            
    
class AdaBoost(object):
    def __init__(self,X,Y,threshold,max_num_classifier=10):
        self.X=X
        self.Y=Y
        self.threshold=threshold
        self.N=len(self.X)
        self.w=[1/self.N for x in range(self.N)]
        self.m=max_num_classifier  #基础阈值分类器最大数目
        self.alpha=[0 for x in range(self.m)]
        
    def calc_w(self,m):
        W=self.w
        Zm=0
        X=self.X
        Y=self.Y
        threshold=self.threshold
        parameters=Parameters(X,Y,W,threshold)
        em=parameters.better_error()
        self.alpha[m]=0.5*log((1-em)/em)
        Gm=[0 for x in range(self.N)]
        fx=[0 for x in range(self.N)]
        for j in range(self.N):
            Gm[j]=parameters.Gxi(self.X[j])
            fx[j]+=self.alpha[m]*Gm[j]
            Zm+=W[j]*exp(-self.alpha[m]*self.Y[j]*Gm[j])
        for j in range(self.N):
            W[j]=W[j]/Zm*exp(-self.alpha[m]*self.Y[j]*Gm[j])
        return W,fx
            
        
    def train(self):
        Fx=np.array([0.0 for x in range(self.N)])
        num=0
        for i in range(self.m):
            self.w,fx=self.calc_w(i)
            Fx+=np.array(fx)
        for j in range(self.N):
            if Fx[j]*self.Y[j]<0:
                num+=1
        return Fx,num,self.w

def main():
    X=[0,1,2,3,4,5,6,7,8,9]
    Y=[1,1,1,-1,-1,-1,1,1,1,-1]
    threshold=[1.5,2.5,3.5,5.5,8.5]
    ada=AdaBoost(X,Y,threshold,max_num_classifier=3)
    print(ada.train())
    
if __name__=="__main__":
    main()





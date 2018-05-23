# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:42:16 2018

@author: wyq

"""   
    

import numpy as np
class DimensionError(ValueError):
    pass

class PCA(object):   #将PCA的参数初始化
    def __init__(self,x,n_component=0):
        self.x=x
        self.dimension=len(x[1])
        if ((n_component==0)or(n_component>=self.dimension)):
            raise DimensionError("the dimension setting is wrong")
        self.n_component=n_component
        
    def Average(self):     #获取原矩阵的均值矩阵，并减去均值矩阵获得新矩阵
        Average_Mat=np.mean(self.x,axis=0)
        NewMat=self.x-Average_Mat
        return NewMat
    
    def cov(self):           #获取新矩阵的协方差矩阵
        NewMat=self.Average()
        covMat=np.cov(NewMat,rowvar=False)
        return covMat
    
    def feature(self):      #获得矩阵的n维特征向量
        covMat=self.cov()
        eVals,eVects=np.linalg.eig(np.mat(covMat))
        eValindice=np.argsort(eVals)#对特征值从小到大排序 
        n=self.n_component
        n_eValindice=eValindice[-1:-(n+1):-1]#最大的n个特征值对应的特征向量
        n_eVects=eVects[:,n_eValindice]
        return n_eVects
    
    def PCA_result(self):    #获取PCA的降维矩阵
        n_eVects=self.feature()
        NewMat=self.Average()
        pca_dimensionMat=NewMat*n_eVects
        return pca_dimensionMat

X=np.array([[2.5,2.4],  
               [0.5,0.7],  
               [2.2,2.9],  
               [1.9,2.2],  
               [3.1,3.0],  
               [2.3,2.7],  
               [2.0,1.6],  
               [1.0,1.1],  
               [1.5,1.6],  
               [1.1,0.9]])  
if __name__=='__main__':
    pca=PCA(X,1)      
    print(pca.PCA_result())
        
    





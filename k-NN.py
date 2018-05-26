# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:55:10 2018

@author: wyq
"""

import numpy as np
import math


class X_Y_train_Error(TypeError):
    pass


class Test_train_Error(TypeError):
    pass


class Train_data_Error(TypeError):
    pass


class KNN(object):
    def __init__(self,X,Y,test_a,K,distance_chosen,label_type):
        self.X=X
        self.Y=Y
        self.test_a=test_a
        self.K=K
        self.distance_chosen=distance_chosen
        self.label_type=label_type
        if(len(self.X)!=len(Y)):
            raise X_Y_train_Error('the train data X and predict label Y do not match' )
        if(len(self.X[0])!=len(self.test_a)):
            raise Test_train_Error('the train data X and test data do not match' )
        for i in range(len(self.X)):
            if(len(self.X[i])!=len(self.X[0])):
                raise Train_data_Error('the Train data does not match')
            
    def normalized(self): #获得训练样本各个属性的最大值减去最小值，为后面的归一化做准备
        Mat=np.array(self.X)
        Mat.sort(axis=0)
        minus_distance=[0 for x in range(len(Mat[0]))]
        for j in range(len(Mat[0])):
            minus_distance[j]=Mat[-1][j]-Mat[0][j]
        return minus_distance
        
    def Manhattan_distance(self):
        train_x=self.X
        test_m=self.test_a
        d_minus=self.normalized()
        D=[0 for x in range(len(train_x))]
        for i in range(len(train_x)):
            for j in range(len(test_m)):
                D[i]=D[i]+(abs(train_x[i][j]-test_m[j])/d_minus[j])
        return D

    def Euclidean_distance(self):
        train_x=self.X
        test_m=self.test_a
        d_minus=self.normalized()
        D=[0 for x in range(len(train_x))]
        d=[0 for x in range(len(train_x))]
        for i in range(len(train_x)):
            for j in range(len(test_m)):
                D[i]=D[i]+((train_x[i][j]-test_m[j])/d_minus[j])**2
        for i in range(len(train_x)):
            d[i]=math.sqrt(D[i])
        return d
    
    def k_sample_choose(self):
        k=self.K
        sample_distance=[0 for x in range(len(self.X))]
        if(self.distance_chosen==1):
            sample_distance=self.Manhattan_distance()
        elif(self.distance_chosen==0):
            sample_distance=self.Euclidean_distance()
        order=np.argsort(sample_distance)
        result=[[0 for col in range(len(self.X))] for row in range(k)]
        i_order=[0 for x in range(k)]
        for i in range(k):
            result[i]=self.X[order[i]]
            i_order[i]=order[i]
        return result,i_order

    def knn_classifier(self):
        y=self.Y
        k=self.K
        label_type=self.label_type
        n_y=len(label_type)
        n_label=[0 for x in range(n_y)]
        r,order=self.k_sample_choose()
        for i in range(k):
            for j in range(n_y):
                if(y[order[i]]==label_type[j]):
                    n_label[j]+=1
        new_label_order=np.argsort(n_label)
        result=label_type[new_label_order[-1]]
        return result
        
X=np.array([[7,8],[10,11],[12,9],[6,3],[9,2]])
Y=[-1,-1,1,1,-1]
test_a=[5,1] 
label_type=[1,-1]
K=3  
distance_chosen=0     
if __name__=='__main__':
    knn=KNN(X,Y,test_a,K,distance_chosen,label_type)
    print(knn.knn_classifier())               
                
        
        

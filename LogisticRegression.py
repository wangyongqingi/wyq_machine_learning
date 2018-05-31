# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:50:20 2018

@author: wyq

"""

import numpy as np
import random

def sigmoid(x):
    result=1/(1+np.exp(-x))
    return result

class X_Y_Match_Error(TypeError):
    pass


class Iteration_ValueError(ValueError):
    pass


class Logistic_Regression(object):
    def __init__(self,X,Y,learning_rate,n_iteration):
        self.X=X
        self.Y=Y
        self.learning_rate=learning_rate
        self.n_iteration=n_iteration
        if(len(self.X)!=len(self.Y)):
            raise X_Y_Match_Error('the train data and label does not match')
        if(type(n_iteration)!=int):
            raise Iteration_ValueError('the type of n_iteration should be int')
    
    def parameter_initialize(self):
        n_features=len(self.X[0])
        w=np.zeros((n_features,1))
        b=0
        parameters=np.insert(w,0,b,axis=0)
        return parameters 
        
    def parameter_adjustment(self):
        X=self.X
        Y=self.Y
        parameters=self.parameter_initialize()
        n_samples=len(X)
        m_features=len(X[0])
        self.parameter_initialize()
        x=np.insert(X,0,1,axis=1)
        y=np.reshape(Y,(n_samples,1))
        for i in range(self.n_iteration):
            j=random.randint(0,(n_samples-1))
            h_x=np.dot(x[j],parameters)
            y_predict=sigmoid(h_x)
            #parameters_ascent=np.dot(x[j].T,(y_predict-y[j]))*self.learning_rate
            parameters_ascent=np.array(x[j].T*(y_predict-y[j])*self.learning_rate)
            parameters_ascent_T=np.reshape(parameters_ascent,(m_features+1,1))
            parameters=parameters+parameters_ascent_T
        return parameters
            
    def classifier(self,x):
        parameters=self.parameter_adjustment()
        cla_x=np.insert(x,0,1,axis=0)
        h_x=np.dot(cla_x,parameters)
        y_predict=sigmoid(h_x)
        y=0
        if(y_predict>=0.5):
            y=1
        elif(y_predict<0.5):
            y=0
        return y

def main():
    data=[[1,2,3,4,5,6],[3,4,5,4,5,6],[5,6,7,4,5,6],[5,8,9,4,5,6],[8,9,5,4,5,6]]
    y=[0,0,0,1,1]
    lr=Logistic_Regression(data,y,learning_rate=0.1,n_iteration=100)
    a=np.array([5,6,7,4,5,6])
    print(lr.classifier(a))
    
if __name__=='__main__':
    main()
            
        
            
            
        
        

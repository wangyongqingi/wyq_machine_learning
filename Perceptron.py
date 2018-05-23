# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:57:31 2018

@author: wyq
"""

import numpy as np

class Learning_rate_Error(ValueError):
    pass


class Prediction_Label_Error(ValueError):
    pass


class Train_data_Error(ValueError):
    pass


class My_classifier_x_Error(ValueError):
    pass


class My_Y_Error(ValueError):
    pass


class Perceptron(object):
    def __init__(self,X,Y,learning_rate,my_classifier_x):
        self.X=X  #X为预测的各个数组的集合，为一个array
        self.Y=Y
        self.learning_rate=learning_rate
        self.my_classifier_x=my_classifier_x
        if(self.learning_rate<=0):
            raise Learning_rate_Error("the learning_rate should be defined larger than 0")
        if(len(my_classifier_x)!=len(self.X[0])):
            raise My_classifier_x_Error('the lenth of my_classifier_x does not match the lenth of X')
        if(len(self.X)!=len(self.Y)):
            raise My_Y_Error('the lenth of My_y does not match the lenth of X')  
        for i in range(len(self.Y)):
            if(abs(self.Y[i])!=1):
                raise Prediction_Label_Error('the prediction label is wrong')
        for i in range(len(self.X)):
            if(len(self.X[i])!=len(self.X[0])):
                raise Train_data_Error('the train data is not match')
        
    def Weight_Bias_Initialize(self):#初始化W，b
        W=[0 for x in range(len(self.X[0]))]
        b=0
        return W,b
    
    def Classifier(self,W,b,Array_x,Array_y):
        negative_number=0
        count_i=0
        for i in range(len(Array_x)):
            if(((np.dot(W,Array_x[i])+b)*Array_y[i])<0):
                negative_number+=1
                count_i=i
            else:
                negative_number+=0
        if(negative_number>0):
            return 1,count_i
        if(negative_number==0):
            return 0,count_i
            
    def W_b_fit(self):
        W,b=self.Weight_Bias_Initialize()
        my_X=self.X
        my_Y=self.Y
        classify=1
        while(classify>0):
            classify,i=self.Classifier(W,b,my_X,my_Y)
            W=self.learning_rate*my_Y[i]*my_X[i]+W
            b=self.learning_rate*my_Y[i]+b
        return W,b
            
    def Perceptron_classifier(self):
        W,b=self.W_b_fit()
        my_predict_x=self.my_classifier_x
        if((np.dot(W,my_predict_x)+b)>0):
            return 1
        elif((np.dot(W,my_predict_x)+b)<0):
            return -1
    
        
    
if __name__=='__main__':
    X= np.array([[3, 3], [4, 3], [1, 1]]) 
    Y = np.array([1, 1, -1]) 
    xo=[1,2]
    per=Perceptron(X,Y,learning_rate=1,my_classifier_x=xo)
    print(per.Perceptron_classifier())
    print(per.W_b_fit())


            
        
        
        
    
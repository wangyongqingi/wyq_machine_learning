# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:41:58 2018

@author: wyq
"""

import numpy as np
class Data_Match_Error(ValueError):
    pass


class Test_data_dimension_Error(ValueError):
    pass


class Naive_Bayesian(object):
    def __init__(self,X1,X2,Y,class_X1,class_X2,class_Y):#class_X1 means all the sorts of X1,class_X2 means all the sorts of X2
    #class_Y means all the sorts of Y
        self.X1=X1
        self.X2=X2
        self.Y=Y
        self.class_X1=class_X1
        self.class_X2=class_X2
        self.class_Y=class_Y
        if((len(self.X1)!=len(self.X2)) or(len(self.X1)!=len(self.Y))):
            raise Data_Match_Error('the data does not match')
    
    def Probablity_of_Y(self):#the function gets the result of all the P(Y)
        m=len(self.Y)
        n=len(self.class_Y)
        y=self.Y
        count=[0 for x in range(n)]
        Probablity=[0 for x in range(n)]
        for i in range(m):
            for j in range(n):
                if(y[i]==self.class_Y[j]):
                    count[j]+=1
        for i in range(n):
            Probablity[i]=count[i]/m
        return Probablity
                    
    def count_type_Y(self):
        m=len(self.Y)
        n=len(self.class_Y)       
        y=self.Y
        class_y=self.class_Y
        count_type_y=[0 for x in range(n)]
        for i in range(m):
            for j in range(n):
                if(y[i]==class_y[j]):
                    count_type_y[j]+=1
        return count_type_y
    
    def Probablity_of_X1_Y(self):
        count_type_y=self.count_type_Y()
        a=len(self.X1)
        b=len(self.Y)
        c=len(self.class_X1)
        d=len(self.class_Y)
        y=self.Y
        x1=self.X1
        class_x1=self.class_X1
        class_y=self.class_Y
        count=[[0 for col in range(c)] for row in range(d)]#以Y为行，以X1为列
        Probablity=[[0 for col in range(c)] for row in range(d)]
        for i in range(a):
            for k in range(c):
                for l in range(d):
                    if((x1[i]==class_x1[k]) and (y[i]==class_y[l])):
                        count[l][k]+=1
                        Probablity[l][k]=count[l][k]/count_type_y[l]
        return Probablity
        
    def Probablity_of_X2_Y(self):
        count_type_y=self.count_type_Y()
        a=len(self.X1)
        c=len(self.class_X2)
        d=len(self.class_Y)
        y=self.Y
        x2=self.X2
        class_x2=self.class_X2
        class_y=self.class_Y
        count=[[0 for col in range(c)] for row in range(d)]#以Y为行，以X1为列
        Probablity=[[0 for col in range(c)] for row in range(d)]
        for i in range(a):
            for k in range(c):
                for l in range(d):
                    if((x2[i]==class_x2[k]) and (y[i]==class_y[l])):
                        count[l][k]+=1
                        Probablity[l][k]=count[l][k]/count_type_y[l]
        return Probablity
    
    def the_order_of_test_data(self,test_data):
        if(len(test_data)!=2):
            raise Test_data_dimension_Error('the dimension of test data is not 2')
        d=len(self.class_X1)
        e=len(self.class_X2)
        a=0
        b=0
        for i in range(d):
            if(test_data[0]==self.class_X1[i]):
                a=i
        for i in range(e):
            if(test_data[1]==self.class_X2[i]):
                b=i
        return a,b
    
    def naive_bayesian_result(self,test_data):
        m,n=self.the_order_of_test_data(test_data)
        probablity_of_Y=self.Probablity_of_Y()
        probablity_of_X1_Y=self.Probablity_of_X1_Y()
        probablity_of_X2_Y=self.Probablity_of_X2_Y()
        a=len(self.class_Y)
        result=0
        order=[0 for x in range(a)]
        final_probablity=[0 for x in range(a)]
        for i in range(a):
            final_probablity[i]=(probablity_of_Y[i]*probablity_of_X1_Y[i][m]*probablity_of_X2_Y[i][n])
        order=np.argsort(final_probablity)
        result=self.class_Y[order[-1]]#最大概率类别
        return final_probablity,result
    
if __name__=='__main__':
    X1=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
    X2=['s','m','m','s','s','s','m','m','l','l','l','m','m','l','l']
    Y=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    class_X1=[1,2,3]
    class_X2=['s','m','l']
    class_Y=[-1,1]
    test_data=[2,'s']
    naive_bayesian=Naive_Bayesian(X1,X2,Y,class_X1,class_X2,class_Y)
    print(naive_bayesian.naive_bayesian_result(test_data))
    
            
        
        
        
        
        
        
             
        
                
        
    
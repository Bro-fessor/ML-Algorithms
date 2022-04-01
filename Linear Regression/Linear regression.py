#!/usr/bin/env python
# coding: utf-8


def batchGradientDescent(fileName,learning_Rate,threshold):
    import numpy as np;
    import sys;
    my_data = np.genfromtxt(fileName, delimiter=',')
    length = len(my_data)
    colcount=len(my_data[0])
    y=np.empty(length,dtype=float)
    #creating matrix for all values of x i.e x0,x1,x2.....xn
    xval=np.empty([length,colcount],dtype=float)
    for i in range(length):
        for j in range(colcount-1):
            xval[i][j+1]= my_data[i][j]
    for i in range(length):
        xval[i][0]=1
    #creating matrix for values of y
    for i in range(length):
        y[i]=(my_data[i][2])
    weights=np.zeros(colcount,dtype=float)
    prev_Square_Error = sys.maxsize
    initial_SSE=np.zeros(length,dtype=float)
    some = 0
    iteration =0
    while(some == 0):
        iteration+=1
        #calculating f(X)
        fx = [0.0 for i in range(length)]
        for i in range(length):
            tempfx=0.0
            for j in range(colcount):
                tempfx=tempfx+xval[i][j]*weights[j]
            fx[i]=tempfx
        #calculating the difference and sum of squared error
        Squared_Error_Sum = 0.0
        for i in range(length):
            initial_SSE[i] = y[i] - fx[i]
            Squared_Error_Sum = Squared_Error_Sum + pow(initial_SSE[i],2)
        print(iteration,' '.join(map(str,weights)),Squared_Error_Sum)
        #calculating gradient
        gradient = np.zeros(colcount,dtype=float)
        for x in range(colcount):
            for i in range(length):
                gradient[x]=gradient[x]+(xval[i][x]*initial_SSE[i])
        #updating weights
        for i in range(colcount):
            weights[i] = weights[i]+(learning_Rate*gradient[i])
        #checking for threshold
        if (prev_Square_Error-Squared_Error_Sum > threshold):
            prev_Square_Error=Squared_Error_Sum
        else:
            some = 1
   



#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
def sigmoid(fileName,eta,iterations):
    
    my_data=np.genfromtxt(fileName, delimiter=',')
    
    #creating a,b and target arrays with values
    length = len(my_data)
    a=np.empty(length,dtype=float)
    b=np.empty(length,dtype=float)
    t=np.empty(length,dtype=float)
    for i in range(length):
        a[i]=(my_data[i][0])
    for i in range(length):
        b[i]=(my_data[i][1])
    for i in range(length):
        t[i]=(my_data[i][2])
    
    #weight initialisation
    w11=0.2
    wa1=-0.3
    wb1=0.4
    w12=-0.5
    wa2=-0.1
    wb2=-0.4
    w13=0.3
    wa3=0.2
    wb3=0.1
    w1o=-0.1
    wh1o=0.1
    wh2o=0.3
    wh3o=-0.4
    
    #printing initial values
    print('-','-','-','-','-','-','-','-','-','-','-',format(w11,'.5f'),format(wa1,'.5f'),format(wb1,'.5f'),format(w12,'.5f'),format(wa2,'.5f'),format(wb2,'.5f'),format(w13,'.5f'),format(wa3,'.5f'),format(wb3,'.5f'),format(w1o,'.5f'),format(wh1o,'.5f'),format(wh2o,'.5f'),format(wh3o,'.5f'))
    
    for x in range(iterations):
        for i in range(length):
            #input to node h1,h2 and h3
            net_h1=(1*w11)+(a[i]*wa1)+(b[i]*wb1)
            net_h2=(1*w12)+(a[i]*wa2)+(b[i]*wb2)
            net_h3=(1*w13)+(a[i]*wa3)+(b[i]*wb3)
            #output of nodes h1, h2 and h3 after passing through activation function
            out_h1=1/(1+np.exp(-net_h1))
            out_h2=1/(1+np.exp(-net_h2))
            out_h3=1/(1+np.exp(-net_h3))
            #input to node o
            net_o=(1*w1o)+(out_h1*wh1o)+(out_h2*wh2o)+(out_h3*wh3o)
            #output of node o after passing through activation function
            final_out=1/(1+np.exp(-net_o))
            #calculating error
            error=0.5*(pow((t[i]-final_out),2))
            if error!=0:
                #back propogation
                delta_o=final_out*(1-final_out)*(t[i]-final_out)
                delta_h1=out_h1*(1-out_h1)*(wh1o*delta_o)
                delta_h2=out_h2*(1-out_h2)*(wh2o*delta_o)
                delta_h3=out_h3*(1-out_h3)*(wh3o*delta_o)
                
                #updating weights
                wh1o=wh1o+(eta*delta_o*out_h1)
                wh2o=wh2o+(eta*delta_o*out_h2)
                wh3o=wh3o+(eta*delta_o*out_h3)
                w1o=w1o+(eta*delta_o*1)
                
                wa1=wa1+(eta*delta_h1*a[i])
                wa2=wa2+(eta*delta_h2*a[i])
                wa3=wa3+(eta*delta_h3*a[i])
                
                wb1=wb1+(eta*delta_h1*b[i])
                wb2=wb2+(eta*delta_h2*b[i])
                wb3=wb3+(eta*delta_h3*b[i])
                
                w11=w11+(eta*delta_h1*1)
                w12=w12+(eta*delta_h2*1)
                w13=w13+(eta*delta_h3*1)
            
            print(format(a[i],'.5f'),format(b[i],'.5f'),format(out_h1,'.5f'),format(out_h2,'.5f'),format(out_h3,'.5f'),format(final_out,'.5f'),format(t[i],'.5f'),format(delta_h1,'.5f'),format(delta_h2,'.5f'),format(delta_h3,'.5f'),format(delta_o,'.5f'),format(w11,'.5f'),format(wa1,'.5f'),format(wb1,'.5f'),format(w12,'.5f'),format(wa2,'.5f'),format(wb2,'.5f'),format(w13,'.5f'),format(wa3,'.5f'),format(wb3,'.5f'),format(w1o,'.5f'),format(wh1o,'.5f'),format(wh2o,'.5f'),format(wh3o,'.5f'))
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    filePath, learningRate, iterations = args.data, args.eta, args.iterations    #Reading the arguments
    sigmoid(filePath,learningRate,iterations)

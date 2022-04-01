
"""
Created on Fri Dec 13 01:28:45 2019
ML Exercise Group 7 11-13 Tutor : Anirban Saha
@author: Aditya Dey, Joel John Philip, Priyanka Byahatti
"""

import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data")
args=parser.parse_args()
filename = 'C:/OvGU_DKe/Machine Learning/Programming Assignment/Assignment 3/Example.tsv'

#import data
data = pd.read_table(filename,header=None,index_col = False)
Ncol = len(data.columns)
print(data)
#remove nan 
data = data.iloc[:,0:Ncol-1]
Ncol = len(data.columns)
#print stats data
data.shape[1]
 def gausscalc(x, m, v):
   product = 1.0
   for i in range(1, len(x)):
      #print(x[i])
      product = product *  (1/np.sqrt(2*3.14*v))* np.exp(-0.5* pow((x[i] - m),2)/v)
   return product      

x = data[0].unique()
c = 0
psf = np.ones((2,2))
for i in x:
   list1 = list()
   df1 = data[data[0]==i]
   for j in range(1,Ncol):
      print(df1[j].mean())
      print(df1[j].var())
      #print(df1[j])
      res = (gausscalc(df1[j].to_numpy(),df1[j].mean(),df1[j].var() ))
      print(res)
    #  for k in range(1,len(df1)):
    #    print (df1[j][k])
   print(len(df1)/len(data))
   #output = np.concatenate((output,list1),axis = 1)

data[1][2]
df1 = data[data[0]=='A']
df1.head()
df1[:1].var()
data[3][2]

print(data.head())
mn1 = list()
mn2 = list()
v1 = list()
v2 = list()
pp = list()
for j in x:
   df = data[data[0] == j]
   for i in range(0,len(df)):
      mn1.append(df[1].mean())
      mn2.append(df[2].mean())
      v1.append(df[1].var())
      v2.append(df[2].var())
      pp.append(len(df)/len(data))
      
len(v1)
data['mn1'] = mn1
data['mn2'] = mn2
data['v1'] = v1
data['v2'] = v2
data['pp'] = pp
gp1yes = list()
gp1no = list()
gp2yes = list()
gp2no = list()
for i in range(0,len(data)):
   gp1yes.append(pp[i] *  (1/np.sqrt(2*3.14*v1[i]))* np.exp(-0.5* pow((data[1][i] - mn1[i]),2)/v1[i]))
   gp1no.append(pp[i] *  (1/np.sqrt(2*3.14*v2[i]))* np.exp(-0.5* pow((data[1][i] - mn2[i]),2)/v2[i]))
   gp2yes.append(pp[i] *  (1/np.sqrt(2*3.14*v1[i]))* np.exp(-0.5* pow((data[2][i] - mn1[i]),2)/v1[i]))
   gp2no.append(pp[i] *  (1/np.sqrt(2*3.14*v2[i]))* np.exp(-0.5* pow((data[2][i] - mn2[i]),2)/v2[i]))
   
data['gp1yes'] = gp1yes
data['gp1no'] = gp1no
data['gp2yes'] = gp2yes
data['gp2no'] = gp2no
data[1][0]

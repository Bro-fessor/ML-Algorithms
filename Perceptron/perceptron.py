import numpy as np
import pandas as pd
import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument('--learningRate', type=float)
parser.add_argument('--threshold', type=float)
args=parser.parse_args()
filename = 'C:/OvGU_DKe/Machine Learning/Programming Assignment/Assignment 3/Example.tsv'

#reading the data into a table
data = pd.read_table(filename,header=None,index_col = False)
print(data)
N = len(data)
Ncol = len(data.columns)
data = data.iloc[:,0:Ncol-1]
# replacing classes with 1's and 0's for easier processing
data[0] = (data[0].replace('A',1))
data[0] = (data[0].replace('B',0))
#shuffling columns to have the class label at the end
cols = list(data.columns)
print(cols[0])
cols = cols[1:] + [cols[0]]
data = data[cols]
data.columns = range(0,Ncol-1)
Ncol = len(data.columns)
#data initialisation
list1 =  list()
list2 = list()
learningRate1 = learningRate2 = 1
str1 = str2 = ""
w1 = np.zeros((Ncol))
gcalw1 = np.zeros((Ncol))
w2 = np.zeros((Ncol))
gcalw2 = np.zeros((Ncol))
count = 1
outfile = filename + "output"
ypred1 = ypred2 =0
error1 = error2 = 0
data = data.to_numpy()
#fd = open(outfile,'a')
while count <= 101 :
    
    esum = 0
    error1 = error2 = 0
    for i in range(0,Ncol):
        gcalw1[i] = 0
        gcalw2[i] = 0
    for i in range (0,N):
       # print(data[i][Ncol-1])
        ypred1 = ypred2 = 0
        ypred1 = ypred1 + w1[0] + np.dot(w1[1:],data[i][:Ncol-1])
        ypred2 = ypred2+ w2[0] + np.dot(w2[1:],data[i][:Ncol-1])
        #with normal learning rate
        if(ypred1 > 0):
           ypred1 = 1
        else:
           ypred1 = 0
        if(ypred1 != data[i][Ncol-1]):
           error1 = error1 + 1
           #print(gcalw, data[i][:Ncol-1])
           #gcalw = gcalw + (learningRate * (data[i][Ncol-1] - ypred) * (data[i][:Ncol-1]))
           gcalw1[0] = gcalw1[0] + learningRate1 * (data[i][Ncol-1] - ypred1)
           gcalw1[1:Ncol] = gcalw1[1:Ncol] + learningRate1 * (data[i][Ncol-1] - ypred1) * (data[i][:Ncol-1])
        # annealing learning rate   
        if(ypred2 > 0):
           ypred2 = 1
        else:
           ypred2 = 0
        if(ypred2 != data[i][Ncol-1]):
           error2 = error2 + 1
           #print(gcalw, data[i][:Ncol-1])
           #gcalw = gcalw + (learningRate * (data[i][Ncol-1] - ypred) * (data[i][:Ncol-1]))
           gcalw2[0] = gcalw2[0] + learningRate2 * (data[i][Ncol-1] - ypred2)
           gcalw2[1:Ncol] = gcalw2[1:Ncol] + learningRate2 * (data[i][Ncol-1] - ypred2) * (data[i][:Ncol-1])   
           
    
    str1 = str1 + str(error1)  
    str2 = str2 + str(error2) 
    list1.append(error1)
    list2.append(error2)
    #fd.write(str2)
    w1 = w1 + gcalw1
    w2 = w2 + gcalw2
    count=count + 1
    #print(learningRate2)
    learningRate2 = 1 / count
    
    
print(str1)
print(str2)
list1
list_between = list("\n")
with open(outfile, 'wt') as out_file:
   tsv_writer = csv.writer(out_file, delimiter='\t')  
   tsv_writer.writerow(list1)
   tsv_writer.writerow(list_between)
   tsv_writer.writerow(list2)
out_file.close()

import tsv
writer = tsv.TsvWriter(open("C:/OvGU_DKe/Machine Learning/Programming Assignment/Assignment 3/Exampleoutput.tsv", "w"))
writer.list_line(list1)
writer.list_line(list2)
writer.close()
#end
#fd.close()
gcalw[1:Ncol-1]
"""gcalw[1:Ncol] + """
"""gcalw[0] + """
for i in range (0,N):
   ypred = ypred + w[0] + np.dot(w[1:],data[i][:Ncol-1])
   print(data[i][:Ncol-2])
   gcalw = gcalw + (learningRate * (data[i][Ncol-1] - ypred) * (data[i][:Ncol-2]))
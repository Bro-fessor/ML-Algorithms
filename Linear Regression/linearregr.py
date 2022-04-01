import numpy as np
import pandas as pd
import argparse
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument('--learningRate', type=float)
parser.add_argument('--threshold', type=float)
args=parser.parse_args()
filename = args.data
learningRate = args.learningRate
threshold = args.threshold
#Read data
data = pd.read_csv(filename,header=None)
N = len(data)
Ncol = len(data.columns)
w = np.zeros((Ncol))
gcalw = np.zeros((Ncol))
count = 0
outfile = "result"+ filename
ypred=0
esumprev = 0
data = data.to_numpy()
fd = open(outfile,'a')
while True :
    str1 = str2 = ""
    esum = 0
    for i in range(0,Ncol):
        gcalw[i] = 0
    #Gradient and Prediction calculation
    for i in range (0,N): 
        ypred = ypred + w[0] + np.dot(w[1:],data[i][:Ncol-1]) 
        error = data[i][Ncol-1] - ypred  
        esum = esum + error*error
        ypred = 0
        gcalw[0] = gcalw[0] + error
        gcalw[1:] = gcalw[1:] + data[i][:Ncol-1] * error
    #Creating Final String    
    for i in range (0,Ncol):
        str1 = str1 + str(round(w[i],4)) + "," 
    str2 = str(count) + "," + str1 + str(round(esum,4)) + "\n"
    print(str2)
    #Write string to file
    fd.write(str2)
    w = w + gcalw * learningRate
    count=count+1
    if( abs(esum - esumprev) <= threshold):
        break
    esumprev = esum
fd.close()



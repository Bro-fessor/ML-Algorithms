import numpy as np
import csv
import math
import argparse
def NaiveBayes(filename):
    data=np.genfromtxt(filename, delimiter=',',dtype=None)

    length=len(data)
    na=0
    nb=0
    for i in range(length):
        if data[i][0].decode('utf-8')=='A':
            na+=1      
        else:
            nb+=1

    #creating lists for x1 and x2 for classes A and B                
    oneA=np.empty(na,dtype=float)
    twoA=np.empty(na,dtype=float)
    oneB=np.empty(nb,dtype=float)
    twoB=np.empty(nb,dtype=float)
    for i in range(length):
        if data[i][0].decode('utf-8')=='A':
            oneA[i]=data[i][1]
            twoA[i]=data[i][2]
        else:
            oneB[i-na]=data[i][1]
            twoB[i-na]=data[i][2]

    #calculating mu values for classes A and B for attributes  x1 and x2
    x1a=0
    x2a=0
    x1b=0
    x2b=0
    for i in range(na) :
        x1a+=oneA[i]
        x2a+=twoA[i]
    m1a=(1/na)*x1a
    m2a=(1/na)*x2a
    for i in range(nb) :
        x1b+=oneB[i]
        x2b+=twoB[i]
    m1b=(1/nb)*x1b
    m2b=(1/nb)*x2b

    #calculating sigma square values for classes A and B for attributes x1 and x2
    y1a=0
    y2a=0
    y1b=0
    y2b=0
    for i in range(na):
        y1a+=pow((oneA[i]-m1a),2)
        y2a+=pow((twoA[i]-m2a),2)
    s1a=(1/(na-1))*y1a
    s2a=(1/(na-1))*y2a
    for i in range(nb):
        y1b+=pow((oneB[i]-m1b),2)
        y2b+=pow((twoB[i]-m2b),2)
    s1b=(1/(nb-1))*y1b
    s2b=(1/(nb-1))*y2b

    #calculating probability of class A and B
    pa=(na/(na+nb))
    pb=(nb/(na+nb))

    print(m1a,s1a,m2a,s2a,pa)
    print(m1b,s1b,m2b,s2b,pb)

    one=np.empty(length,dtype=float)
    two=np.empty(length,dtype=float)
    for i in range(length):
        one[i]=data[i][1]
        two[i]=data[i][2]

    #instances belonging to class A but misclassified as B
    AA=0
    AB=0
    for i in range(na):
        px1a=(1/math.sqrt(2*3.14*s1a))*math.exp(-0.5*pow((one[i]-m1a),2)/s1a)
        px2a=(1/math.sqrt(2*3.14*s2a))*math.exp(-0.5*pow((two[i]-m2a),2)/s2a)
        px1b=(1/math.sqrt(2*3.14*s1b))*math.exp(-0.5*pow((one[i]-m1b),2)/s1b)
        px2b=(1/math.sqrt(2*3.14*s2b))*math.exp(-0.5*pow((two[i]-m2b),2)/s2b)
        res1=(px1a*px2a*pa)/((px1a*px2a*pa)+(px1b*px2b*pb))
        res2=(px1b*px2b*pb)/((px1a*px2a*pa)+(px1b*px2b*pb))
        if res1>res2:
            AA+=1
        else:
            AB+=1

    #instances belonging to class B but misclassified as A
    BA=0
    BB=0
    for i in range(na+1,length):
        px1a=(1/math.sqrt(2*3.14*s1a))*math.exp(-0.5*pow((one[i]-m1a),2)/s1a)
        px2a=(1/math.sqrt(2*3.14*s2a))*math.exp(-0.5*pow((two[i]-m2a),2)/s2a)
        px1b=(1/math.sqrt(2*3.14*s1b))*math.exp(-0.5*pow((one[i]-m1b),2)/s1b)
        px2b=(1/math.sqrt(2*3.14*s2b))*math.exp(-0.5*pow((two[i]-m2b),2)/s2b)
        res1=(px1a*px2a*pa)/((px1a*px2a*pa)+(px1b*px2b*pb))
        res2=(px1b*px2b*pb)/((px1a*px2a*pa)+(px1b*px2b*pb))
        if res1>res2:
            BA+=1
        else:
            BB+=1

    #Claculating number of Missclassifications
    Misclas=AB+BA
    print(Misclas)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    args = parser.parse_args()
    filePath = args.data    #Reading the arguments
    NaiveBayes(filePath)

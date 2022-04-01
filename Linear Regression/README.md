# Linear-Regression
Python 3 program

A program that implements a (batch) linear regression using the gradient descent method using the following gradient calculation:

<img width="246" alt="Formula" src="https://user-images.githubusercontent.com/56769691/116580015-98ae0d80-a930-11eb-9b8e-e979fc37e8f1.png">

where xi is one data point, N is the size of the data set, η is the learning rate, yi is the target output and f(xi) is the linear function given by f(x) = transpose(w)* x or f(x) = Σ(wi · xi), w and x include the bias, i.e. w0 and x0 is always equal to 1. All weights are initialized to 0. Two random example data sets  named random1 and random2 as csv files are used as input. Using these data sets, the gradient descent method is implemented and for each iteration the weights and sum of squared errors are printed until a given threshold of change in the error is reached.

The output looks like: 
iteration_number,weight0,weight1,weight2,...,weightN,sum_of_squared_errors

The solution is rounded to 4 decimals and for the data sets, learning rate of 0.0001 and a threshold of 0.0001 is considered. Program accepts the following parameters:

1. threshold - the change in error that has to fall below, before the algorithm terminates
2. data - the location of the data file (ex: media/data/records.csv)
3. eta - the learning rate of the gradient descent 


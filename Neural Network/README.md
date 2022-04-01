# Neural-Network
Python 3

A program is written that implements a simple sigmoid neural network using the delta rule using the following activation function:

<img width="309" alt="Formula" src="https://user-images.githubusercontent.com/56769691/116667277-a360b500-a9b9-11eb-9ad3-992630c4022f.png">

where w is the vector of weights including the bias w0, all attributes and weights are treated as double-precision values. Two data sets named Gauss3 and Gauss4 as csv files are provided. and the program reads both data sets and treats the last value of each line as the class (1 being positive and 0 being negative). Implementation of the back propagation algorithm with single sample correction is done, which means weights are updated for every data point. For that purpose, a fixed architecture is used for ease and is as given in the figure below.

<img width="240" alt="Network" src="https://user-images.githubusercontent.com/56769691/116667418-cc814580-a9b9-11eb-82aa-67796a71905c.png">

The learning rate is η = 0.2 and 2 full iterations are carried out, which means going through all data points twice. The initialization is hardcoded. Machine learning libraries are not used and the program accepts the following parameters:

1. data - the location of the data file (ex: media/data/report.csv)

2. eta - the learning rate for the backpropagation

3. iterations - the number of iterations to calculate

The output format :

delta_h1 represent the error of h1 and h1 the output of h1 itself, same is true for other nodes. The first line indicates the initialization for the weights and rest of the lines represent the weights after updating and the output and error for that iteration along with the original data point.




Network train and text classes.

The neural network (Main.java) (NN) takes in 784 parameters (from 28x28 pixel image). It has 2 hidden
layers and 10 outputs which correspond to numbers from 0 to 9. After every 5000 iterations of minibatches 
of 10 it calls the NetworkTest class which tests the network using the test samples. It should finally 
stop when the guess probability is over 90% but I actually never got to that but instead to around 87%.
The script doesn't exactly use a test data set but instead relies on the validation data set. The NN 
uses backpropagation to optimize the weights and biases.

Some optimizing could be done regarding the speed of the network training but I'll leave it to anyone
who wants to use the script. For example, standardizing the data might not be necessary and even
normalizing could be enough or maybe even not doing anything to it. Standardizing of the whole data
could also be done beforehand so the script wouldn't need to standardize the data during each iteration.

I used the MNIST data for the NN.

Improved version of the number recognition neural network.

It divides the figure pixel inputs into 4 groups, upper left hand corner (0), upper right hand corner (1),
lower left hand corner (2) and lower right hand corner (3). It passes these inputs to a single hidden
layer which has been divided into 4 groups (0 to 3) of 10 neurons each (40 total) lowering the amount of total
connections by a factor of 4. These 40 neurons are then connected to the ouput layer of 10 neurons 
corresponding to numbers from 0 to 9.

It converges a lot faster and after 20000 iterations of minibatches of 10 I managed to reach
over 90% guess rate.

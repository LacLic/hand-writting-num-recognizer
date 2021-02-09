""" !README!

Code Style:

    Class: ThisIsAnExample()
    ClassMethod: ThisIsAnExample.thisIsAnotherExample()
    NormalFunction: this_is_the_third_example()
    NormalVarible: this_is_the_forth_example
    Const: THIS_IS_A_CONST

"""


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# (?) ensure the plots are inside this notebook, not an external window
# %matplotlib inline


class neuralNetwork():  # neural network class definition
    # initialize the neural network

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

        # set number of nodes in each input, hidden, output layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # learning rate
        self.learnRate = learningRate

        # link weight matrices, wih and who
        # weight inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(
            0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = numpy.random.normal(
            0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # annoymous func
        self.actiFunc = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):

        # convert inputs and targets list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.actiFunc(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.actiFunc(final_inputs)

        # error is the difference of target and output
        output_errors = targets - final_inputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.learnRate * \
            numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                      numpy.transpose(hidden_outputs))
        """
        self.who += self.learnRate * \
            numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                      hidden_outputs.T)"""
        # between the input and hidden layers
        self.wih += self.learnRate * \
            numpy.dot((hidden_errors*hidden_outputs *
                       (1.0 - hidden_outputs)), numpy.transpose(inputs))

    # query the neural network

    def query(self, inputs_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.actiFunc(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.actiFunc(final_inputs)

        return final_outputs


input_nodes = 784  # 28*28
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

with open('train.csv', 'r') as data_file:
    data_list = data_file.readlines()[1:]  # 除首行外，全部读取到内存中
    # print(data_list)
    data_file.close()
# data_file = open('train.csv', 'r')
# data_list = data_file.readlines()
# data_file.close()

# print(data_list[0])

# # debug: check whether the img is read successfully
# all_values = data_list[1].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# """
# numpy.asfarray() turn string into real number
# .reshape() make it into 28x28 matrix
# """
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# matplotlib.pyplot.imsave('temp.png', image_array, cmap='Greys')
# # ^~~~debug::end

for record in data_list:
    all_values = record.split(',')
    # make data into 0.01 ~ 1.00
    """ Remains problem
    (why?)
    0.01: prevent 0.00 leading to failing to update the weight
    1.00: just preventing the output value being 1.00 is enough
    """
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    # numpy.zeros(): use 0 to pad the array
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] is the label of the record
    targets[int(all_values[0])] = 0.99
    network.train(inputs, targets)

# load the test data CSV file into a list
with open('test.csv', 'r') as test_data_file:
    test_data_list = test_data_file.readlines()[1:]
    test_data_file.close()

# from test.csv
test = test_data_list[0]
test_values = test.split(',')
test_inputs = (numpy.asfarray(test_values)/255.0 * 0.99) + 0.01
print(network.query(test_inputs))
# print(test_data_list[0].strip().split(','))

# from train.csv
test = data_list[999]
print('Answer is', test[0])
test_values = test.split(',')
test_inputs = (numpy.asfarray(test_values[1:])/255.0 * 0.99) + 0.01
print(network.query(test_inputs))

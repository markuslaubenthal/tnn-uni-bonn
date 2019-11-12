# Authors: Markus Laubenthal, Lennard Alms



import numpy as np
# Some Activation functions

import sys

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
n_layers = (1 + len(sys.argv))
arg_dimensions = []
if len(sys.argv) >= 2:
    for arg in range(1, len(sys.argv)):
        arg_dimensions.append(int(sys.argv[arg]))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ident(x):
    return x

eta = 0.001

# np.random.seed(1)

def readfile(filename):
    dimensions = []
    input_data = []
    output_data = []
    line_count = 1

    with open(filename) as file:
        for line in file:
            line = line.replace('\t', '')
            line = line.replace('  ', ' ')
            line = line.replace('  ', ' ')
            line = line.replace('  ', ' ')
            line = line.replace('  ', ' ')
            line = line.replace('  ', ' ')
            line = line.replace('  ', ' ')
            line = line.strip()
            if(line_count == 2):
                line = line.strip("# \t")
                tmp = line.split(" ")
                for val in tmp:
                    if(not val.startswith("P")):
                        val = val[2:]
                        dimensions.append(np.int(val))
            elif (line.startswith('0') or line.startswith('1') or line.startswith('-') or line.startswith('+')):
                # input, output = line.split(' ')
                linedata = line.split(' ')
                input = linedata[:dimensions[0]]
                output = linedata[dimensions[0]:dimensions[0] + dimensions[1]]
                # print(input)
                # print(output)
                input_data.append(np.array(input))
                output_data.append(np.array(output))
                # input_data.append(np.array(('1 ' + input).split(' ')))
                # output_data.append(np.array(output.split(' ')))
            line_count += 1
    return np.array(input_data).astype(np.float64), np.array(output_data).astype(np.float64), dimensions

training_data, label_data, layer_dimensions = readfile("PA-B-train-01.dat")


if len(arg_dimensions) > 0:
    _in = layer_dimensions[0]
    _out = layer_dimensions[1]
    layer_dimensions = np.append(_in, arg_dimensions)
    layer_dimensions = np.append(layer_dimensions, _out)
# Initialize layer dimensions
# layer_dimensions = [4, 3, 5, 5]

# Initialize Weight Matrices for every layer
weight_matrices = []
for i in range(n_layers - 1):
    w = (np.random.rand(layer_dimensions[i + 1], layer_dimensions[i] + 1) - 0.5) * 4
    weight_matrices.append(w)


# Initialize a list wich will contain the output of each layer
layer_output = []
for i in range(n_layers):
    layer_output.append(np.zeros(layer_dimensions[i]))

# Initializing Deltas
delta = []
for i in range(n_layers):
    delta.append(np.zeros(layer_dimensions[i]))

# Setup Transferfunctions for each Layer
# By default all layers will be initialized with the sigmoid function
# Functions can be overwritten after initialization with
# layer_transfer_functions[layer_number] = function_name
# with 0 <= layer_number < n_layers
layer_transfer_functions = [ident]
for i in range(1, n_layers):

    layer_transfer_functions.append(tanh)

def feedforward(input):
    global layer_output
    global weight_matrices
    layer_output[0] = layer_transfer_functions[0](input)
    # layer_output[0] = input
    output = None
    for i in range(1, n_layers):
        layer_input = np.append([1], layer_output[i - 1])
        layer_output[i] = layer_transfer_functions[i](weight_matrices[i - 1].dot(layer_input))
        output = layer_output[i]
    return output

def isOutputNeuron(layer_index):
    if layer_index == n_layers - 1:
        return True
    return False

def calculateDelta(layer_index):
    global delta
    global layer_output
    global teacher
    # TODO: Replace static derivation with dynamic derivation
    # derivative = layer_output[layer_index] * (1 - layer_output[layer_index])
    derivative = 1 - layer_output[layer_index] ** 2
    if(isOutputNeuron(layer_index)):
        delta[layer_index] = (teacher - layer_output[layer_index]) * derivative
    else:
        weight_matrix = weight_matrices[layer_index]
        weight_matrix = np.transpose(weight_matrix)
        weight_matrix = weight_matrix[1:]

        # weight_matrix = np.transpose(weight_matrices[layer_index])[1:]
        _delta = (delta[layer_index + 1].reshape(delta[layer_index + 1].shape[0], 1))
        delta[layer_index] = np.dot(weight_matrix, _delta).flatten() * derivative

def squaredError(expected, output):
    diff = expected - output
    diff_squared = np.power(diff, 2)
    sum = diff_squared.sum()
    return sum

# Total Squared Error Function over all Patterns
def totalError():
    err = 0
    for i, pattern in enumerate(training_data):
        err += squaredError(label_data[i], feedforward(pattern))
    return err / len(training_data)

def train(n_iterations):
    global eta
    global training_data
    global weight_matrices
    global layer_output
    global teacher
    for iteration in range(n_iterations):
        for pattern_index, pattern in enumerate(np.random.permutation(training_data)):
            teacher = label_data[pattern_index]
            feedforward(pattern)
            # maximum n_iterations
                # Iterate through each layer, starting at Output Layer

            for layer_index in range(1,n_layers)[::-1]:
                calculateDelta(layer_index)
                weight_matrix = weight_matrices[layer_index - 1]
                # Foreach neuron in current layer
                for neuron_index in range(weight_matrix.shape[0]):
                    # Foreach weight connected to the neuron
                    for weight_index in range(weight_matrix.shape[1]):
                        weight = weight_matrix[neuron_index][weight_index]
                        output = None
                        if(weight_index == 0):
                            output = 1
                        else:
                            output = layer_output[layer_index - 1][weight_index - 1]
                        delta_w = eta * delta[layer_index][neuron_index] * output
                        weight_matrix[neuron_index][weight_index] += delta_w
                        weight_matrices[layer_index - 1] = weight_matrix
            if(iteration % 300 == 0):
                print(totalError())
teacher = None
print(feedforward(training_data[0]))
print(feedforward(training_data[1]))
print(feedforward(training_data[2]))
print(feedforward(training_data[3]))
train(1000000)

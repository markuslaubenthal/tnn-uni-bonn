import numpy as np
# Some Activation functions

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ident(x):
    return x

eta = 0.001

np.random.seed(1)

def readfile(filename):
    dimensions = []
    input_data = []
    output_data = []
    line_count = 1

    with open(filename) as file:
        for line in file:
            line = line.strip()
            if(line_count == 2):
                line = line.strip("# \t")
                tmp = line.split("    ")
                for val in tmp:
                    if(not val.startswith("P")):
                        val = val[2:]
                        dimensions.append(np.int(val))
            elif (line.startswith('0') or line.startswith('1')):
                input, output = line.split('  ')
                input_data.append(np.array(('1 ' + input).split(' ')))
                output_data.append(np.array(output.split(' ')))
            line_count += 1
    return np.array(input_data)[:,:-1].astype(np.float64), np.array(output_data).astype(np.float64), dimensions

training_data, label_data, layer_dimensions = readfile("PA-B-train-01.dat")
# Initialize layer dimensions
# layer_dimensions = [4, 3, 5, 5]
n_layers = len(layer_dimensions)

# Initialize Weight Matrices for every layer
weight_matrices = []
for i in range(n_layers - 1):
    weight_matrices.append((np.random.rand(layer_dimensions[i + 1], layer_dimensions[i] + 1) - 0.5) * 4)

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
    layer_transfer_functions.append(sigmoid)

def feedforward(input):
    global layer_output
    global weight_matrices
    layer_output[0] = layer_transfer_functions[0](input)
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
    global label_data
    # TODO: Replace static derivation with dynamic derivation
    derivative = layer_output[layer_index] * (1 - layer_output[layer_index])
    if(isOutputNeuron(layer_index)):
        delta[layer_index] = (label_data - layer_output[layer_index]) * derivative
    else:
        delta[layer_index] = np.dot(np.transpose(weight_matrices[layer_index-1]), delta[layer_index + 1]) * derivative
        x = 1


def train(n_iterations):
    global eta
    global training_data
    global weight_matrices
    global layer_output
    for pattern in np.random.permutation(training_data):
        feedforward(pattern)
        # maximum n_iterations
        for iteration in range(n_iterations):
            # Iterate through each layer, starting at Output Layer
            for layer_index in range(n_layers)[::-1]:
                calculateDelta(layer_index)
                weight_matrix = weight_matrices[layer_index - 1]
                # Foreach neuron in current layer
                for neuron_index in range(weight_matrix.shape[0]):
                    # Foreach weight connected to the neuron
                    for weight_index in range(weight_matrix.shape[1]):
                        weight = weight_matrix[neuron_index][weight_index]
                        print("-------")
                        print(layer_index)
                        print(neuron_index)
                        print(weight_index)
                        delta_w = eta * delta[layer_index][neuron_index] * layer_output[layer_index - 1][weight_index]
                        weight_matrix[neuron_index][weight_index] += delta_w
print(feedforward(training_data[0]))
train(10)
print(feedforward(training_data[0]))

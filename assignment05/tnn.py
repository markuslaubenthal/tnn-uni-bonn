# Authors Markus Laubenthal, Lennard Alms

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3456789)


# Read the file
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


sigma, N, H, M, centers, weightsHM = None, None, None, None, None, None
eta = None
gauss_output = None
training_data, label_data = None, None

# Initialize all important values
def init(_N = 2, _H = 4, _M = 1, _eta = 0.01):
    global sigma
    global N
    global H
    global M
    global centers
    global weightsHM
    global eta
    global training_data
    global label_data

    N = _N
    H = _H
    M = _M
    eta = _eta

    sigma = np.random.rand(H) * 2
    weightsHM = np.random.rand(M, H) - 0.5

    # Initialize the RBF Centers
    # Create a vector for every neuron with zeros
    # Then For every Neuron set them uniformly on the input space
    centers = np.zeros((_H, N))
    for i in range(_N):
        data_min = training_data[:,i].min()
        data_max = training_data[:,i].max()
        centers[:,i] = np.linspace(data_min, data_max, _H)



# Gauss Function as activation function for rbf neurons
def gauss_bell(c, x, sigma):
    global gauss_output
    gauss_output = np.exp(-np.linalg.norm(c-x, axis = 1) / (2 * sigma ** 2))
    return gauss_output

# Calculate 1 Feed Forward Step
def feedforward(c, weights, x, sigma):
    return np.dot(weights, gauss_bell(c, x, sigma))



training_data, label_data, dimensions = readfile('train.dat')

init(_N = dimensions[0], _H = 9, _M = dimensions[1], _eta = 0.01)


# Do the gradient descent
def gradient_descent(centers, weights, sigma):
    error = []
    global H, M, N, eta

    # For n_iterations
    for it in range(50000):
        # Pick a pattern
        for index, pattern in enumerate(training_data):
            # Get the Label and Calculate the output
            label = label_data[index]
            output = feedforward(centers, weights, pattern, sigma)
            # Iterate over Output Neurons
            for output_neuron_index in range(M):
                # Calculate the difference between Label and output
                difference = label[output_neuron_index] - output[output_neuron_index]
                # Calculate the deltaError (Derivative of Error)
                deltaError = - difference * gauss_output
                for rbf_neuron_index in range(H):
                    # For Every weight apply the error for the neuron and multiply it
                    # with the learning rate
                    # then add delta_w to the weight
                    delta_w = deltaError[rbf_neuron_index] * eta
                    weights[output_neuron_index][rbf_neuron_index] -= delta_w
        if(it % 50 == 0):
            error.append(totalError())

    return error

def squaredError(expected, output):
    diff = expected - output
    diff_squared = np.power(diff, 2)
    sum = diff_squared.sum()
    return sum

def totalError():
    err = 0
    global training_data
    global label_data
    global centers
    global weightsHM
    global sigma
    for i, pattern in enumerate(training_data):
        err += squaredError(label_data[i], feedforward(centers, weightsHM, pattern, sigma))
    return err / len(training_data)


error = gradient_descent(centers, weightsHM, sigma)

plt.plot(error)
plt.show()

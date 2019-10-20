import numpy as np
import sys

# Initial Variable Definitions
n, m = None, None
weights = None
fermi_res = None
training_data = None
training_label = None
expected = None

init_testing_dataset = True

# Functions reads a file and returns the Training-Data and Label Data
def readfile(filename):
    input_data = []
    output_data = []
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if (line.startswith('0') or line.startswith('1')):
                input, output = line.split('\t')
                input_data.append(np.array(('1 ' + input).split(' ')))
                output_data.append(np.array(output.split(' ')))
    return np.array(input_data)[:,:-1].astype(np.uint8), np.array(output_data).astype(np.uint8)


# If no arguments are given, we will read from File PA_A-train.dat
# and use its data to perform the backpropagation
if not len(sys.argv) > 2:
    print("File PA_A-train.dat will be loaded...")
    training_data, training_label = readfile('PA-A-train.dat')
    n = training_data.shape[1] - 1
    m = training_label.shape[1]
    init_testing_dataset = False

# Initialize random weight values between -0.5 and 0.5
# in a weight matrix with m rows and n+1 columns
weights = np.random.rand(m, n + 1) - 0.5

# fermi_res will contain the last result of every neuron after the feedforward
fermi_res = np.zeros(m)
expected = np.zeros(m)

# Set Max Iterations and Learning Rate eta
iterations = 100000
learning_rate = 0.1

# Feed Forward is a matrix Multiplication with the weights between input and output layer
# and the calculation of the transferfunction on the output vector
def feedforward(vector):
    global weights
    global fermi_res
    result = weights.dot(vector)
    fermi_res = transferfunction(result)
    return fermi_res

# Transfer Function (Fermi Function)
def transferfunction(vector):
    return 1.0 / (np.exp(-vector) + 1)

# Squared Error Function to calculate the error of all neurons
def squaredError(expected, output):
    diff = expected - output
    diff_squared = np.power(diff, 2)
    sum = diff_squared.sum()
    return sum

# Total Squared Error Function over all Patterns
def totalError():
    err = 0
    for i, pattern in enumerate(training_data):
        err += squaredError(training_label[i], feedforward(pattern))
    return err / len(training_data)



# m is the neuron, that we inspect. n is the weight that we inspect
def updateDeltaWForLastPattern(m, n, pattern):
    global expected
    global fermi_res
    global learning_rate
    global weights
    # Delta Rule
    deltaw = learning_rate * (expected[m] - fermi_res[m]) * fermi_res[m] * (1 - fermi_res[m]) * pattern[n]
    weights[m][n] += deltaw


def train():
    global expected
    iteration = 0
    total_error = 1000
    error_threshold = 0.003
    # Train for set max iterations or until error is smaller or equal to error_threshold
    while(iteration < iterations and total_error > error_threshold):
        # Shuffle the Training Data and iterate over them
        for index in np.random.permutation(range(training_data.shape[0])):
            # Specify Pattern and expected output
            inputvector = training_data[index]
            expected = training_label[index]
            # Calculate the Output of the Network
            output = feedforward(inputvector)
            # Calculate the squared Error for that specific pattern
            squared_error = squaredError(expected, output)
            # Iterate over all output Neurons
            for o in range(m):
                # Iterate over all weights that correspond to the neuron
                for w in range(n + 1):
                    # Apply Delta rule to weight
                    updateDeltaWForLastPattern(o, w, inputvector)
        # Keep Track of Error over Training Steps
        if(iteration % 300 == 0):
            total_error = totalError()
            print(total_error)
        iteration += 1

# Execute main Function
train()

# Print the results
print("Rounded Predictions on Test Data:")
for i, pattern in enumerate(training_data):
    print("Result: ", np.round(feedforward(pattern)).astype(np.uint8), " Expected: ", training_label[i])

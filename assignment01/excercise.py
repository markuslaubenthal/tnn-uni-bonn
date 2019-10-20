import numpy as np
import sys
import matplotlib.pyplot as plt

n, m = None, None
weights = None
fermi_res = None
training_data = None
training_label = None
expected = None

init_testing_dataset = True

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



if not len(sys.argv) > 2:
    print("File PA_A-train.dat will be loaded...")
    training_data, training_label = readfile('PA-A-train.dat')
    n = training_data.shape[1] - 1
    m = training_label.shape[1]
    init_testing_dataset = False


else:
    if not sys.argv[1].isdigit():
        print("First argument must be a number.")
        exit()
    if not sys.argv[2].isdigit():
        print("Second argument must be a number.")
        exit()
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    training_data = np.zeros((8, n + 1))
    training_label = np.zeros((8, 1))

weights = np.random.rand(m, n + 1) - 0.5
fermi_res = np.zeros(m)
expected = np.zeros(m)








iterations = 100000
learning_rate = 0.1

# Initialize Weights with n+1 rows and m columns


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

def createDataset():
    global training_data
    global training_label
    training_data = np.array([
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]
    ])
    training_label = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])

def squaredError(expected, output):
    expected_vec = np.full(output.shape, expected)
    diff = expected_vec - output
    diff_squared = np.power(diff, 2)
    sum = diff_squared.sum()
    return sum

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
    deltaw = learning_rate * (expected[m] - fermi_res[m]) * fermi_res[m] * (1 - fermi_res[m]) * pattern[n]
    weights[m][n] += deltaw

def train():
    global expected
    iteration = 0
    total_error = 1000
    while(iteration < iterations and total_error > 0.003):
        for index in np.random.permutation(range(training_data.shape[0])):
            inputvector = training_data[index]
            expected = training_label[index]
            output = feedforward(inputvector)
            squared_error = squaredError(expected, output)
            for o in range(m):
                for w in range(n + 1):
                    updateDeltaWForLastPattern(o, w, inputvector)
        if(iteration % 300 == 0):
            total_error = totalError()
            print(total_error)
        iteration += 1

if(init_testing_dataset):
    createDataset()

train()
print("Rounded Predictions on Test Data:")
for i, pattern in enumerate(training_data):
    print("Result: ", np.round(feedforward(pattern)).astype(np.uint8), " Expected: ", training_label[i])

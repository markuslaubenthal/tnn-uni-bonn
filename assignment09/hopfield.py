# authors: Markus Laubenthal, Lennard Alms

import numpy as np
import time

# Set parameters here
my_theta = 5
n_neurons = 24

# Set patterns to learn and test here
# if not set, random patterns will be created

# 3 patterns, with 24 parameters
# learning_patterns = np.array([
#     [1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
#     [1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1],
#     [1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1],
# ])

# test_patterns = np.array([
#     [1,1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
#     [1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1],
#     [1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,1,1,-1],
# ])

learning_patterns = None
test_patterns = None

np.random.seed(10)

def createPatterns(amount, k = 10):
    p = np.random.randint(2, size=(amount, k))
    p[p == 0] = -1
    return p

if learning_patterns is None:
    learning_patterns = createPatterns(10, k=n_neurons)
if test_patterns is None:
    test_patterns = createPatterns(10, k=n_neurons)

def train(patterns):
    weight_matrix = patterns.transpose() @ patterns - np.identity(patterns.shape[1]) * patterns.shape[0]
    return weight_matrix

def energy(weight_matrix, pattern, theta):
    return -1/2 * np.sum(
            weight_matrix * (pattern[:,None] @ pattern.reshape(1, pattern.shape[0]))
        ) + np.sum(theta * pattern)

def printState(weight_matrix, pattern, theta = None):
    if(pattern.shape[0] < 101):
        for p in pattern:
            if p == 1:
                print("+",end = '')
            else:
                print("-",end = '')
        print("")
    elif(theta is not None):
        print("Error:", energy(weight_matrix, pattern, theta))
    else:
        print("No Theta given")


def retreive(pattern, weight_matrix, theta):
    input = pattern.copy()

    iteration = 0
    converged = False

    print("Input:")
    printState(weight_matrix, input, theta)
    print("Steps:")

    while not converged:
        iteration += 1
        input_old = input.copy()

        for p in np.random.permutation(range(pattern.shape[0])):
            sum = np.sum(weight_matrix[:,p] * input)
            if(sum > theta[p]):
                input[p] = 1
            else:
                input[p] = -1
        if(np.array_equal(input, input_old)):
            converged = True
        printState(weight_matrix, input, theta)
    print("Converged after", iteration, "steps")

w_m = train(learning_patterns)

theta = np.full((learning_patterns.shape[1]), my_theta)

for pattern in test_patterns:
    print("")
    retreive(pattern, w_m, theta)
    print("")

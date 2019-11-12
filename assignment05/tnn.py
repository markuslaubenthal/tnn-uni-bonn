import numpy as np

# np.random.seed(3456789)

sigma, N, H, M, centers, weightsHM = None, None, None, None, None, None
eta = None
gauss_output = None

def init(_N = 2, _H = 4, _M = 1, _eta = 0.01):
    global sigma
    global N
    global H
    global M
    global centers
    global weightsHM
    global eta
    N = _N
    H = _H
    M = _M
    eta = _eta

    sigma = np.random.rand(H) * 2
    weightsHM = np.random.rand(M, H) - 0.5
    centers = np.random.rand(H, N) - 0.5



def gauss_bell(c, x, sigma):
    global gauss_output
    gauss_output = np.exp(-np.linalg.norm(c-x, axis = 1) / (2 * sigma ** 2))
    return gauss_output

def feedforward(c, weights, x, sigma):
    return np.dot(weights, gauss_bell(c, x, sigma))


learning_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

label_data = np.array([
    [-0.2],
    [0.2],
    [0.2],
    [-0.2]
])

init(_N = 2, _H = 4, _M = 1, _eta = 0.01)
# override
centers = np.array([[0,0], [0,1], [1,0], [1,1]])

def gradient_descent(centers, weights, sigma):
    global H, M, N, eta
    for it in range(50000):
        for index, pattern in enumerate(learning_data):
            label = label_data[index]
            output = feedforward(centers, weights, pattern, sigma)
            for output_neuron_index in range(M):
                difference = label[output_neuron_index] - output[output_neuron_index]
                deltaError = - difference * gauss_output
                for rbf_neuron_index in range(H):
                    delta_w = deltaError[rbf_neuron_index] * eta
                    weights[output_neuron_index][rbf_neuron_index] -= delta_w

    return None

input = np.array([0,1])
print(feedforward(centers, weightsHM, np.array([0,0]), sigma))
print(feedforward(centers, weightsHM, np.array([0,1]), sigma))
print(feedforward(centers, weightsHM, np.array([1,0]), sigma))
print(feedforward(centers, weightsHM, np.array([1,1]), sigma))

gradient_descent(centers, weightsHM, sigma)

print(feedforward(centers, weightsHM, np.array([0,0]), sigma))
print(feedforward(centers, weightsHM, np.array([0,1]), sigma))
print(feedforward(centers, weightsHM, np.array([1,0]), sigma))
print(feedforward(centers, weightsHM, np.array([1,1]), sigma))

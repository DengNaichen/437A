import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sin, square
from dnn_units import *
import copy



N = 128
theta = np.arange (0, 2 * np.pi, np.pi/(N/2))
matrix = np.zeros((N, N))
for i in range (0,N):
    for j in range (0,N):
        matrix [i][j] = np.abs(np.sin((theta[j] - theta [i])))


def free_energy (AL, rho, matrix):

    """

    This function for calculate cost
    Y -- The output of model
    cost -- The free energy, F tilda

    """
    m = AL.shape[1]

    # todo
    cost = (1./m) * 2*pi * np.dot(AL,np.log(rho * AL).T) + 4 * pi**2 * rho *  np.dot (np.dot(AL, matrix), AL.T)

    cost = np.squeeze(cost)
    assert (cost.shape == () )

    return cost 



def relu_backward_free_e(dA, cache):
    """

    no need to change this with loss function
    Z: from cache
    dA: change with loss function
    for Z > 0, Z' = 1, so dZ = dA
    for Z <= 0, Z' = 0, so dZ = dA
    
    """
    Z = cache  

    dZ = np.array(dA, copy=True) 

    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

### no need to modify this function

def linear_backward_free_e(dZ, cache):
    """

    this function describes the last third equation 
    no need to change with loss fucntion

    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward_free_e(dA, cache):

    linear_cache, activation_cache = cache
    
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)  # the 
    
    return dA_prev, dW, db

def L_model_backward_free_e(AL, caches, rho, matrix):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]

    # need to change this line to corrsponding with my loss function
    dAL = np.log (AL + 1) + (4 * pi** 2 * rho/m**2) * np.dot(AL, matrix) 

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "relu")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def L_layer_model_free_e (rho, X, matrix, layers_dims, learning_rate = 0.06, num_iterations= 50000):

    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)
        
        cost = free_energy (AL, rho)

        grads = L_model_backward_free_e(AL, caches, rho)

        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def model_optimize_free_e (X, rho, matrix, input_parameters, layers_dims, optimizer, learning_rate = 0.0007, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                
    t = 0                            # initializing the counter required for Adam update
    m = X.shape[1]                   # number of training examples

    parameters = copy.deepcopy(input_parameters)
    v, s = initialize_adam(parameters)
    
    for i in range(num_epochs):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        cost = free_energy (AL, rho, matrix)

        grads = L_model_backward_free_e(AL, caches, rho, matrix)

        # Update parameters
        t = t + 1 # Adam counter
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
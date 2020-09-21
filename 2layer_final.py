import numpy as np
import matplotlib.pylab as plt

"""
loss function: 
compute_cost = Mean Square Error Loss
compute_cost_2 = cross entropy function
two layers
gradient decent

"""

step_size = 2.0*np.pi/200.0
X = np.arange(0.0, 2.0*np.pi, step_size).reshape((1,200))
Y = np.sin(X)**2



x_test = X

def layer_sizes (X,Y):
    n_x = X.shape[0]
    n_h = 16
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters (n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*np.sqrt(1/n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*np.sqrt(1/n_h)
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                 "b1" :b1,
                 "W2": W2, 
                 "b2":b2}
    return parameters

def sigmoid (x):
    return 1.0/(1.0+np.exp(-x))


def forward_propagation(X, parameters):
    # retrieve paramters from dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2 
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    diff = (A2 - Y)**2
    cost = (1.0/Y.shape[1])*np.sum(diff)
    cost = float(np.squeeze(cost))
    return cost

def compute_cost_entropy(A2, Y,parameters):
    
    m = Y.shape[1]


    cost = -(1/m)*np.sum(np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)) 
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]

    W1 = parameters ["W1"]
    W2 = parameters ['W2']

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y  ## here is the most important part I need to change!!!!
    dW2 = 1.0/m*np.dot(dZ2, A1.T)  # I don't need to change this line
    db2 = 1.0/m*np.sum(dZ2, axis=1, keepdims=True)     # keep this line
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))        # but I don't understand!
    dW1 = 1.0/m*np.dot(dZ1, X.T)
    db1 = 1.0/m*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.01):  

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
 
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

cost_points = []
iteration_points = []

def nn_model(X, Y, n_h, num_iterations = 50000, print_cost=False):

    n_x = layer_sizes (X, Y)[0]
    n_y = layer_sizes (X, Y)[2]

    parameters = initialize_parameters (n_x, n_h, n_y) 
    for i in range (0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost_entropy(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y) 
        parameters = update_parameters(parameters, grads, learning_rate = 0.6)
        if print_cost and i % 10000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            cost_points.append(cost)
            iteration_points.append(i)
    return parameters


parameters = nn_model(X, Y, 16, num_iterations=500000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

plt.plot(iteration_points,cost_points)
plt.xlabel ("iteration")
plt.ylabel ("cost")
plt.show()

def predict (parameters, x_test):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    z1 = np.dot(W1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2 
    a2 = sigmoid(z2)
    y_predict = a2
    return y_predict

y_predict = predict (parameters, x_test)

plt.plot(x_test.T, y_predict.T)
plt.plot(X.T, Y.T)
plt.show()
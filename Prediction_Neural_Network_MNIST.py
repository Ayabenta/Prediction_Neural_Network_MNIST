#!/usr/bin/env python
# coding: utf-8

# Fait par : El Keddadi Kaoutar et Bentaleb Aya

# In[52]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils


# In[53]:


def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    
    L_out : int
        Number of outgoing connections. 
    
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
        
    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    """

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================


    # ============================================================
    return W


# In[54]:


initial_Theta1 = randInitializeWeights(400, 18)
initial_Theta2 = randInitializeWeights(18,18)
initial_Theta3 = randInitializeWeights(18, 10)
nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel(),initial_Theta3.ravel()], axis=0)


# In[55]:


def sigmoidGradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z. 
    This should work regardless if z is a matrix or a vector. 
    In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    
    Parameters
    ----------
    z : array_like
        A vector or matrix as input to the sigmoid function. 
    
    Returns
    --------
    g : array_like
        Gradient of the sigmoid function. Has the same shape as z. 
    
    Instructions
    ------------
    Compute the gradient of the sigmoid function evaluated at
    each value of z (z can be a matrix, vector or scalar).
    
    Note
    ----
    We have provided an implementation of the sigmoid function 
    in `utils.py` file accompanying this assignment.
    """

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = utils.sigmoid(z) * (1 - utils.sigmoid(z))

    # =============================================================
    return g


# In[136]:


# 20x20 Input Images of Digits
input_layer_size  = 400

hidden_layer_size = 18
num_labels = 10

#  training data stored in arrays X, y
data = loadmat(os.path.join(r'Data', 'ex4data1.mat'))

X, y = data['X'], data['y'].ravel()


y[y == 10] = 0
m = y.size
d=np.concatenate((X,y.reshape(5000,1)), axis=1)
data=np.split(d,20,axis=0)
data_test=data[12]
data_etude=np.concatenate((data[0],data[1],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[13],data[14],data[15],data[16],data[17],data[18],data[19]), axis=0)
data_etude.shape

X = data_etude[:,:400]
y_f = data_etude[:,-1].reshape(4500)
y=y[:4500]
layers=[400,18,18,10]


# In[123]:





# In[137]:


def nnCostFunction(nn_params,Layers,X, y, lambda_=0.0):
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:Layers[1] * (Layers[0] + 1)],(Layers[1], (Layers[0] + 1)))
    nn_params = nn_params[Layers[1] * (Layers[0] + 1):]
    Theta2 = np.reshape(nn_params[:Layers[2] * (Layers[1] + 1)],(Layers[2], (Layers[1] + 1)))
    Theta3 = np.reshape(nn_params[Layers[2] * (Layers[1] + 1):],(Layers[3], (Layers[2] + 1)))
    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    Theta2_grad = np.zeros(Theta3.shape)
    
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = utils.sigmoid(a1.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    a3 = utils.sigmoid(a2.dot(Theta2.T))
    a3 = np.concatenate([np.ones((a3.shape[0], 1)), a3], axis=1)
    
    a4 = utils.sigmoid(a3.dot(Theta3.T))
    
    y_matrix = np.eye(Layers[-1])[y]
    p=np.argmax(a4, axis=1)
    print('%.2f' % (np.mean(p== y) * 100))
    temp1 = Theta1
    temp2 = Theta2
    temp3 = Theta3
    
    # Add regularization term
    
    reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])) + + np.sum(np.square(temp3[:, 1:])))
    
    J = (-1 / m) * np.sum((np.log(a4) * y_matrix) + np.log(1 - a4) * (1 - y_matrix)) + reg_term
    
    # Backpropogation
    
    delta_4 = a4 - y_matrix
    delta_3 = delta_4.dot(Theta3)[:, 1:] * sigmoidGradient(a2.dot(Theta2.T))
    delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)
    Delta3 = delta_4.T.dot(a3)
    
    # Add regularization to gradient

    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    
    Theta3_grad = (1 / m) * Delta3
    Theta3_grad[:, 1:] = Theta3_grad[:, 1:] + (lambda_ / m) * Theta3[:, 1:]
    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel(), Theta3_grad.ravel()])

    return J, grad


# In[125]:


lambda_ = 0
J, _ = nnCostFunction(nn_params, layers, X, y, lambda_)
print('Cost at parameters (loaded from ex4weights): %.6f ' % J)


# In[138]:


#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 1000}

#  You should also try different values of lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, layers, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        nn_params,
                        jac=True,
                        method='CG',
                        options=options)

# get the solution of the optimization
nn_params = res.x
print(res)        
# Obtain Theta1 and Theta2 back from nn_params


# In[134]:


J, _ = nnCostFunction(nn_params, layers, X, y, lambda_)
y


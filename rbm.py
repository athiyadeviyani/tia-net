import os
import sys
import time
import math
import random
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#-----------------------------------------------------------------------
# Utility functions, do not modify
#-----------------------------------------------------------------------

if not os.path.exists('../plot'):
    os.makedirs('../plot')
if not os.path.exists('../dump'):
    os.makedirs('../dump')

seed = 10417617

def binary_data(inp):
    # Do not modify
    return (inp > 0.5) * 1.

def sigmoid(x):
    """
    Args:
        x: input
    Returns: the sigmoid of x
    """
    # Do not modify
    return 1 / (1 + np.exp(-x))

def xavier_init(n_input, n_output):
    """
    # Use Xavier weight initialization
    # Xavier Glorot and Yoshua Bengio, 
    "Understanding the difficulty of training deep feedforward neural networks"
    """
    # Do not modify
    b = np.sqrt(6/(n_input + n_output))
    return np.random.normal(0,b,(n_output, n_input))

def shuffle_corpus(X, y=None):
    """shuffle the corpus randomly
    Args:
        X: the image vectors, [num_images, image_dim]
        y: the image digit, [num_images,], optional
    Returns: The same images and digits (if supplied) with different order
    """ 
    # Do not modify
    random_idx = np.random.permutation(len(X))
    if y is None:
        return X[random_idx]
    return X[random_idx], y[random_idx]

# Do not modify ^^^^
#-----------------------------------------------------------------------

class RBM:
    def __init__(self, n_visible, n_hidden, k, lr, max_epochs):
        """The RBM base class
        Args:
            n_visible: Dimension of visible features layer
            n_hidden: Dimension of hidden layer
            k: gibbs sampling steps
            lr: learning rate, remains constant through train
            max_epochs: Number of train epochs
        Returns:
            Instantiated class with following parameters
            hbias: Bias for the hidden layer, shape (n_hidden, )
            vbias: Bias for the visible layer, shape (n_visible, )
            W: Weights between visible and hidden layer, shape (n_visible, n_hidden)
        """
        # Instantiate RBM class constants
        #---------------------------------------------
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = lr
        self.max_epochs = max_epochs
        
        # Initialize hidden and visible biases with zeros
        # Initialize visible weights with Xavier (random_weight_init above)
        # Initialize classification weights with Xavier
        #---------------------------------------------
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)
        self.W     = xavier_init(n_visible, n_hidden)

    def h_v(self, v):
        """ Transform the visible vector to hidden vector and 
            compute its probability being 1
        Args:
            v: Visible vector (n_visible, )
        Returns:
            1. Probability of hidden vector h being 1 p(h=1|v), shape (n_hidden, )
        """
        return sigmoid(np.dot(v, self.W.T) + self.hbias)

    def sample_h(self, h_prob):
        """ 
        Sample a hidden vector given the distribution p(h=1|v)
        
        Args: 
            h_prob: probability vector p(h=1|v), shape (n_hidden, )
        Return:
            1. Sampled hidden vectors, shape (n_hidden, )
        """
        return np.random.binomial(1, h_prob)
        
    def v_h(self, h):
        """
        Transform the hidden vector to visible vector and
            compute its probability being 1
        
        Args:
            h: the hidden vector h (n_hidden,)
        Return:
            Hint: sigmoid provided function.
            1. Probability of output visible vector v being 1, shape (n_visible,)
        """
        return sigmoid(np.dot(h, self.W) + self.vbias)

    def sample_v(self, v_prob, v_true=None, v_observation=None):
        """ 
        Sample a visible vector given the distribution p(v=1|h)
        Args: 
            v_prob: probability vector p(v=1|h), shape (n_visible,)
            v_true: Ground truth vector v, shape (n_visible, )
            v_observation: a 0-1 mask that tells which index is observed by the RBM, 
                    where 1 means observed, and 0 means not observed, shape (n_visible, )
            
            Example:
            Say v is of size (2,), v_true is [1, 0], and the v_observation is [0, 1], 
            then we reveal the second true entry "0" to the RBM.
            
            When you do gibbs sampling, you "inject" the observed part of v_true: 
                      v_true * v_observation
            to the RBM, so that you have a super certain probability distribution,
            on the observed indexes. Here the "*" is entry-wise multiplication.                        
        Return:
            Hint: NumPy binomial sample
            1. Sampled visible vector, binary in our experiment
                shape (n_visible,)
        """
        # if v_true and v_observation:
        #     for i in range(len(v_prob)):
        #         if v_observation[i]:
        #             v_prob[i] = v_true[i]
        # vectorize
        if v_true is not None and v_observation is not None:
            v_prob = np.where(v_observation==1, v_true, v_prob)

        return np.random.binomial(1, v_prob)

    def gibbs_k(self, v, k=0, v_true=None, v_observation=None):
        """ 
        The contrastive divergence k (CD-k) procedure,        
        with the possibility of injecting v_true observed values.
        Args:
            v: the input visible vector (n_visible,)
            v_true: Ground truth vector v, shape (n_visible, )
            v_observation: a 0-1 mask that tells which index is observed by the RBM, 
                    where 1 means observed, and 0 means not observed, shape (n_visible, )
            k: the number of gibbs sampling steps, scalar (int)
        Return:
            Hint: complete the tests and use the methods h_v, sample_h, v_h, sample_v
            1. h0: Hidden vector sample with one iteration (n_hidden,)
            2. v0: Input v (n_visible,)
            3. h_sample: Hidden vector sample with k iterations  (n_hidden,)
            4. v_sample: Visible vector sampled wit k iterations (n_visible,)
            5. h_prob: Prob of hidden being 1 after k iterations (n_hidden,)
            6. v_prob: Prob of visible being 1 after k itersions (n_visible,)
        """
        v0 = binary_data(v)
        h0_prob = self.h_v(v0)
        
        # MY CODE BELOW
        h0 = self.sample_h(h0_prob)
        h_sample = h0
        
        for i in range(k if k > 0 else self.k):
            v_prob = self.v_h(h_sample)
            v_sample = self.sample_v(v_prob, v_true, v_observation)
            h_prob = self.h_v(v_sample)
            h_sample = self.sample_h(h_prob)

        return h0, v0, h_sample, v_sample, h_prob, v_prob

    def update(self, x):
        """ 
        Update the RBM with input v.
        Args:
            v: the input data X , shape (n_visible,)
        Return: self with updated weights and biases
            Hint: Compute all the gradients before updating weights and biases.
        """

        h0, v0, h_sample, v_sample, h_prob, v_prob = self.gibbs_k(x)
       
        # MY CODE BELOW

        """
        CD-k Algorithm
        For each training example x
        1. Generate a negative sample x~ using k steps on Gibbs sampling, starting at the data point x
        2. Update model parameters
            positive_phase = h(x)x.T
            negative_phase = h(x~)x~T
            W <= W + lr * (positive_phase - negative_phase)
            b <= b + lr * (h(x) - h(x~))
            c <= c + lr * (x - x~)
        """

        # b <= b + lr * (h(x) - h(x~))
        hx_prob = self.h_v(x) 
        h_prob = self.h_v(v_sample)
        h_update = hx_prob - h_prob
        self.hbias = self.hbias + self.lr * h_update
        self.hbias = self.hbias.reshape(-1)

        # c <= c + lr * (x - x~)
        v_update = x - v_sample
        self.vbias = self.vbias + self.lr * v_update
        self.vbias = self.vbias.reshape(-1)

        # positive_phase = h(x)x.T
        # negative_phase = h(x~)x~T
        # W <= W + lr * (positive_phase - negative_phase)
        positive_phase = np.dot(hx_prob.reshape(1, -1).T, x.reshape(1, -1))
        negative_phase = np.dot(h_prob.reshape(1, -1).T, v_sample.reshape(1, -1))
        self.W = self.W + self.lr * (positive_phase - negative_phase)

    def get_error(self, x, sampled_x):
        return np.sqrt(np.sum((x - sampled_x)**2))

    def evaluate(self, X, k=0):
        """ 
        Compute reconstruction error
        Args:
            X: the input X, shape (len(X), n_visible)
        Return:
            The reconstruction error, shape a scalar
        """
        N = len(X)
        reconstruction_error = 0
        for x in X:
            h0, v0, h_sample, v_sample, h_prob, v_prob = self.gibbs_k(x, k)
            reconstruction_error += self.get_error(x, v_sample)
        return reconstruction_error / N

    def fit(self, X, valid_X):
        """ 
        Fit RBM, do not modify. Note that you should not use this function for conditional generation.
        Args:
            X: the input X, shape (len(X), n_visible)
            X_valid: the validation X, shape (len(valid_X), n_visible)
        Return: self with trained weights and biases
        """
        # Do not modify
        # Initialize trajectories
        self.loss_curve_train_ = []
        self.loss_curve_valid_ = []

        # Train
        for epoch in tqdm(range(self.max_epochs)):
            shuffled_X = shuffle_corpus(X)
            
            for i in range(len(shuffled_X)):
                x = shuffled_X[i]
                self.update(x)

            # Evaluate
            train_recon_err = self.evaluate(shuffled_X, k=1)
            valid_recon_err = self.evaluate(valid_X, k=1)
            self.loss_curve_train_.append(train_recon_err)
            self.loss_curve_valid_.append(valid_recon_err)
            
            # Print optimization trajectory
            train_error = "{:0.4f}".format(train_recon_err)
            valid_error = "{:0.4f}".format(valid_recon_err)
            print(f"Epoch {epoch+1} :: \t Train Error {train_error} \
                  :: Valid Error {valid_error}")
        print("\n\n")


if __name__ == "__main__":

    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument('-train', type=str, help='training file path', default='./data/digitstrain.txt')
    parser.add_argument('-valid', type=str, help='validation file path', default='./data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path", default="./data/digitstest.txt")
    parser.add_argument('-max_epochs', type=int, help="maximum epochs", default=10)

    parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=250)
    parser.add_argument('-k', type=int, help="CD-k sampling", default=3)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1)

    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]
    train_X = binary_data(train_X)

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_X = binary_data(valid_X)
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_X = binary_data(test_X)
    test_Y = test_data[:, -1]

    n_visible = train_X.shape[1]

    print("input dimension is " + str(n_visible))

    rbm = RBM(n_visible=n_visible, n_hidden=args.n_hidden, 
              k=args.k, lr=args.lr, max_epochs=args.max_epochs)

    rbm.fit(X=train_X, valid_X=valid_X)

    # you can access the train and validation error trajectories
    # from the self.loss_curve_train_ and self.loss_curve_valid_ attributes

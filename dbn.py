import math
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

from rbm import *
from rbm import RBM

from tqdm import tqdm

class DBN:
    def __init__(self, n_visible, layers, k, lr, max_epochs):
        """ 
        The Deep Belief Network (DBN) class
        Args:
            n_visible: Dimension of visible features layer
            layers: a list, the dimension of each hidden layer, e.g,, [500, 784]
            k: gibbs sampling steps
            lr: learning rate, remains constant through train
            max_epochs: Number of train epochs
        """
        # Instantiate DBN class constants
        #---------------------------------------------
        self.n_visible = n_visible
        self.layers = layers
        self.k = k
        self.lr = lr
        self.max_epochs = max_epochs

        # Instantiate RBM components through the layers
        #----------------------------------------------
        self.rbms = []
        rbm = RBM(n_visible=n_visible, n_hidden=layers[0], k=k, lr=lr, max_epochs=max_epochs) # Instantiate the first RBM
        self.rbms.append(rbm)
        for i in range(1, len(self.layers)):
            rbm = RBM(n_visible=layers[i-1], n_hidden=layers[i], k=k, lr=lr, max_epochs=max_epochs)
            self.rbms.append(rbm)

    def fit(self, X, valid_X):
        """ The training process of a DBN, basically we train RBMs one by one
        Args:
            X: the train images, numpy matrix
            valid_X: the valid images, numpy matrix
        """

        # zero lists for reconstruction errors
        self.te_list = np.zeros((len(self.rbms), self.max_epochs))
        self.ve_list = np.zeros((len(self.rbms), self.max_epochs))

        # iterate over all RBMs
        for i in range(len(self.rbms)):
            if i > 0:  # get new data
                train = []
                valid = []
                
                for x in X:
                    h_v = self.rbms[i-1].h_v(x)
                    sample_h = self.rbms[i-1].sample_h(h_v)
                    train.append(sample_h)
                    
                for x in valid_X:
                    h_v = self.rbms[i-1].h_v(x)
                    sample_h = self.rbms[i-1].sample_h(h_v)
                    valid.append(sample_h)
                    
                train = np.array(train)
                valid = np.array(valid)

            else:
                train = X
                valid = valid_X

            # iterate over all epochs
            for epoch in tqdm(range(self.max_epochs)):
                shuff = shuffle_corpus(train)

                for x in shuff:
                    # update the RBM weights
                    self.rbms[i].update(x)

                te = self.rbms[i].evaluate(train)
                ve = self.rbms[i].evaluate(valid)
                self.te_list[i][epoch] = te
                self.ve_list[i][epoch] = ve

                # Print optimization trajectory
                train_error = "{:0.4f}".format(te)
                valid_error = "{:0.4f}".format(ve)
                print(f"Epoch {epoch + 1} :: RBM {i + 1} :: \t " +
                      f"Train Error {te} :: Valid Error {ve}")
            print("\n")

            
def fit_mnist_dbn(n_visible, layers, k, max_epochs, lr):
    train_data = np.genfromtxt('data/digitstrain.txt', delimiter=",")
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]
    train_X = train_X[-900:]

    valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=",")
    valid_X = valid_data[:, :-1][-300:]
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt('data/digitstest.txt', delimiter=",")
    test_X = test_data[:, :-1][-300:]
    test_Y = test_data[:, -1]

    train_X = binary_data(train_X)
    valid_X = binary_data(valid_X)
    test_X = binary_data(test_X)

    n_visible = train_X.shape[1]
    
    dbn = DBN(n_visible=n_visible, layers=layers, 
              k=k, max_epochs=max_epochs, lr=lr)
    dbn.fit(X=train_X, valid_X=valid_X)
    
    return dbn

  
if __name__ == "__main__":
    
    np.seterr(all='raise')
    plt.close('all')

    dbn = fit_mnist_dbn(n_visible=784, layers=[500, 784], k=3, max_epochs=200, lr=0.01)

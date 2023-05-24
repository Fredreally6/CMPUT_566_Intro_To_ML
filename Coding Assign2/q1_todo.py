from utils import plot_data, generate_data
import numpy as np


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

def sig(z):
    return 1/(1+np.exp(-z))  # use sigmoid function as activation function

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # use gradient descent solution
    lr = 0.1                # set learning rate = 0.1
    _ , m = X.shape
    w = np.zeros(m)         # initializing
    b = 0
    for _ in range(10000):  # gradient descent
        y = sig(X @ w + b)
        w = w - lr * (X.T @ (y - t)/m)
        b = b - lr * np.sum((y - t)/m)
    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    t = sig(X @ w + b)
    for i in range(len(t)):
        if t[i] >= 0.5:
            t[i] = 1
        else:
            t[i] = 0
    return t 


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    # use close-form solution
    n, m = X.shape
    w = np.zeros(m)         # initializing
    b = 0
    one = np.ones((n,1))
    X = np.concatenate([X, one], axis=1)
    w = np.linalg.inv(X.T @ X) @ X.T @ t    # (d+1)*1, and here d=1
    w = w[:-1]                
    b = w[-1]
    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    t = X @ w + b
    for i in range(len(t)):
        if t[i] >= 0:
            t[i] = 1
        else:
            t[i] = 0
    return t 



def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    acc = np.sum(t == t_hat)/len(t)
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()

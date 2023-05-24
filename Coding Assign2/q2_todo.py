# -*- coding: utf-8 -*-
from hashlib import sha1
import numpy as np
import struct
import matplotlib.pyplot as plt
import scipy.sparse

def readMNISTdata():
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def getLoss(X, W, t=None):

    y = X @ W
    #softmax method
    zmax = np.max(y, axis=1)    # method from stackoverflow.com/a/39558290
    zmax = zmax[:, np.newaxis]  # add dimension
    zi = y - zmax
    zj = np.sum(np.exp(zi), axis=1)
    zj = zj[:,np.newaxis]

    y_hat = np.exp(zi) / zj
    t_hat = np.argmax(y_hat, axis=1)

    # get loss and gradient, method from gist.github.com/awjuliani/5ce098b4b76244b7a9e3#file-softmax-ipynb
    m = t.shape[0]
    t = t[:,0]
    onehot = scipy.sparse.csr_matrix((np.ones(m),(t, np.array(range(m)))))
    onehot = np.array(onehot.todense()).T
    n = X.shape[0]
    loss = (-1 / n) * np.sum(onehot * np.log(y_hat))
    gradient = (-1 / n) * (X.T @ (onehot - y_hat))

    # get accuracy
    acc = np.sum(t_hat == t.flatten()) / len(t_hat)

    return y, t_hat, loss, acc, gradient

def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    y, t_hat, loss, acc, _ = getLoss(X, W, t)
    return y, t_hat, loss, acc

def train(X_train, y_train, X_val, t_val):
    global MaxEpoch, N_class

    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    w = np.zeros([X_train.shape[1], N_class])

    loss_train = []
    accuracy = []
    W_best = None
    epoch_best = 0
    acc_best = 0

    for epoch in range(MaxEpoch):
        loss_epoch = 0
        for batch in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[batch * batch_size: (batch+1) * batch_size]
            y_batch = y_train[batch * batch_size: (batch+1) * batch_size]

            _, _, loss_batch, _, grad = getLoss(X_batch, w, y_batch)
            loss_epoch = loss_epoch + loss_batch

            w = w - alpha * grad - alpha * decay * w    #gradient descent

        # training loss in epoch, and add it to the losses list
        training_loss = loss_epoch / int(np.ceil(N_train/batch_size))
        loss_train.append(training_loss)

        # validation accuracy in epoch
        _, _, _, acc = predict(X_val, w, t_val)
        accuracy.append(acc)

        # finding the best epoch
        if acc_best <= acc:
            acc_best = acc
            epoch_best = epoch
            W_best = w


    return epoch_best, acc_best, W_best, loss_train, accuracy

# print plots
def plot_losses(loss_train):
    picture = plt.figure()
    plt.plot(loss_train, label = "training loss")
    plt.title("alpha = 0.1, MaxEpoch = 50, batch_size =100")
    plt.legend()
    plt.xlabel("number of epoch")
    plt.ylabel("training loss")
    plt.savefig('loss.png')

def plot_accuracy(acc_val):
    picture = plt.figure()
    plt.plot(acc_val, label = "validation accuracy")
    plt.title("alpha = 0.1, MaxEpoch = 50, batch_size =100")
    plt.legend()
    plt.xlabel("number of epoch")
    plt.ylabel("validation accuracy")
    plt.savefig('accuracy.png')    




##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()



N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best, W_best, loss_train, accuracy = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)

print('Best epoch =', epoch_best, 'best validation accuracy = ', acc_best, 'test accuracy = ',acc_test)

plot_losses(loss_train)
plot_accuracy(accuracy)
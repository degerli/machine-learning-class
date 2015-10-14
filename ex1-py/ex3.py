import numpy as np
import scipy.io


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    # Load weights
    mat = scipy.io.loadmat('../ex3/ex3weights.mat')
    theta1 = mat['Theta1']  # Theta1 has size 25 x 401
    theta2 = mat['Theta2']  # Theta2 has size 10 x 26

    # Load data (input layer) and labels
    mat = scipy.io.loadmat('../ex3/ex3data1.mat')
    y = mat['y'][:, 0] - 1  # true labels as 1D array
    m = mat['X'].shape[0]   # number of training examples
    ones = np.ones((m, 1))  # creating zeroth features
    a1 = np.append(ones, mat['X'], axis=1)  # append bias column (zeroth features)

    # compute layer-2
    z2 = a1.dot(theta1.T)
    z2 = np.append(ones, z2, axis=1)  # append bias column (zeroth features)
    a2 = sigmoid(z2)

    # compute layer-3 (output layer)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)

    # get actual class predictions
    h = a3.argmax(axis=1)

    # compare predictions with expected labels
    accuracy = np.sum(h == y) / m

    # print accuracy
    print('accuracy:', accuracy)


if __name__ == '__main__':
    main()

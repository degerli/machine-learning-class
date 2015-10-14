import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y, sym):
    plt.plot(x, y, sym)
    plt.xlabel('Exam 2 score')
    plt.ylabel('Exam 1 score')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(X, y, theta):
    h = sigmoid(X.dot(theta))
    m = y.shape[0]  # number of training examples

    cost1 = -np.log(h[y == 1])
    cost2 = -np.log(1 - h[y == 0])

    return (np.sum(cost1) + np.sum(cost2)) / m


def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]  # number of training examples
    J_history = []

    for i in range(0, iterations):
        h = sigmoid(X.dot(theta))
        theta -= X.T.dot(h - y) * (1 / m) * alpha
        # print('iteration:', i)
        J_history.append(cost_function(X, y, theta))

    return theta, J_history


def predict(X, theta):
    p = np.round(sigmoid(X.dot(theta)))
    return p;


def accuracy(X, theta, y):
    p = predict(X, theta)
    m = y.shape[0]
    return np.sum(p == y) / m


def plotDecBound(theta, X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plot_data(X[pos][:, 1], X[pos][:, 2], 'k+')
    plot_data(X[neg][:, 1], X[neg][:, 2], 'ko')

    plot_x = [np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]

    print(plot_x)

    plot_y = (-1 / theta[2]) * (theta[2] * plot_x + theta[1])

    plt.plot(plot_x, plot_y)

    plt.show()


def main():
    print('Plotting Data ...\n')
    data = np.loadtxt('../ex2/ex2data1.txt', delimiter=',')

    # pos = np.where(data[:, 2] == 1)
    # neg = np.where(data[:, 2] == 0)
    #
    # plot_data(data[pos][:, 0], data[pos][:, 1], 'k+')
    # plot_data(data[neg][:, 0], data[neg][:, 1], 'ko')
    #
    # plt.show()

    ones = np.ones((data.shape[0], 1))
    data = np.append(ones, data, axis=1)

    cols = data.shape[1]

    X = data[:, 0:(cols - 1)]
    y = data[:, (cols - 1):cols]

    theta = np.zeros((X.shape[1], 1))

    cost = cost_function(X, y, theta)

    print('initial cost:', cost)
    print('initial accuracy:', accuracy(X, theta, y))

    # # Some gradient descent settings
    # iterations = 1000000
    # alpha = 0.003
    # theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    # cost = cost_function(X, y, theta)

    print('cost after gradient descent:', cost)
    print('accuracy after gradient descent:', accuracy(X, theta, y))
    plotDecBound(theta, X, y)

    # print('Plotting the linear fit')
    # plt.plot(X[:, 1:2], X.dot(theta))
    # plt.legend(['Training data', 'Linear regression'], 4 )
    # plt.show()


if __name__ == '__main__':
    main()

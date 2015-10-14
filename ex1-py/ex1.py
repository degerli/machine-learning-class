import numpy as np
import matplotlib.pyplot as plt


def warm_up_exercise():
    return np.identity(5)


def plot_data(x, y):
    plt.plot(x, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    # plt.show()


def compute_cost(X, y, theta):
    m = y.shape[0]  # number of training examples
    h = np.dot(X, theta)
    squared_errors = np.square(h - y)
    error = (1 / (2 * m)) * np.sum(squared_errors)
    return error


def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]  # number of training examples
    J_history = []

    for i in range(0, iterations):
        h = np.dot(X, theta)
        theta -= np.dot(X.T, h - y) * (1 / m) * alpha
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def normal_equation(X, y):
    return np.linalg.pinv(np.dot(X.T, X)).dot(X.T).dot(y)


def main():
    print('Running warmUpExercise ... \n');
    print('5x5 Identity Matrix: \n');
    warm_up_exercise()

    print('Plotting Data ...\n')
    data = np.loadtxt('../ex1/ex1data1.txt', delimiter=',')
    plot_data(data[:, 0], data[:, 1])

    ones = np.ones((data.shape[0], 1))
    data = np.append(ones, data, axis=1)

    X = data[:, 0:2]
    y = data[:, 2:3]

    theta = np.zeros((X.shape[1], 1))

    error = compute_cost(X, y, theta)
    print('initial error:', error)

    print('Plot the linear fit')
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    error = compute_cost(X, y, theta)
    print('error after gradient descent:', error)

    print('Plotting the linear fit')
    plt.plot(X[:, 1:2], X.dot(theta))
    plt.legend(['Training data', 'Linear regression'], 4 )
    plt.show()

    theta = normal_equation(X,y)
    error = compute_cost(X, y, theta)
    print('error after normal equation:', error)







    # plt.xlabel('Number of iterations')
    # plt.ylabel('Cost J')
    # plt.plot(range(0, 50), J_history[0:50])
    #
    # theta = np.zeros((X.shape[1], 1))
    #
    # alpha = 0.001
    # theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    # plt.plot(range(0, 50), J_history[0:50])
    #
    # plt.show()

if __name__ == '__main__':
    main()

import numpy as np

def costfn(X, Theta, Y, num_users, num_movies, num_features, lmbda):

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    X = np.mat(X)
    Theta = np.mat(Theta)

    J = 0.5 * np.power(((X * np.transpose(Theta)) - Y), 2)
    J = sum(sum(np.array(J)))

    grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))
    return J, grad

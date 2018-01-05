import numpy as np

def costfn(params, Y, num_users, num_movies, num_features):

    X = np.array(params[:num_movies * num_features]).reshape(num_movies, num_features)
    Theta = np.array(params[X.size:]).reshape(num_users, num_features)

    # Calculate J(cost)
    J = 0.5 * np.power(((np.mat(X) * np.transpose(np.mat(Theta))) - Y), 2)
    J = sum(sum(np.array(J)))

    return J

def gradfn(params, Y, num_users, num_movies, num_features):

    X = np.array(params[:num_movies * num_features]).reshape(num_movies, num_features)
    Theta = np.array(params[X.size:]).reshape(num_users, num_features)

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # Calculate grad(gradient)
    for i in range(num_movies):
        idx = np.where((Y[i] > 0))
        Theta_temp = Theta[idx, :]
        Y_temp = Y[i, idx]
        if Theta_temp.size > 0:
            X_grad[i, :] = (np.mat(X[i, :]) * np.mat(Theta_temp).transpose() - np.mat(Y_temp)) * np.mat(Theta_temp);

    for j in range(num_users):
        idx = np.where((Y[:, j]))
        X_temp = X[idx, :]
        Y_temp = Y[idx, j]
        if X_temp.size > 0:
            Theta_grad[j, :] = (np.mat(Theta[j, :]) * np.mat(X_temp).transpose() - np.mat(Y_temp)) * np.mat(X_temp)

    grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))
    return grad
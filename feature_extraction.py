import numpy as np
from preprocess import preprocess
from normalize_ratings import normalize
import cofi_costfn
import scipy

# Plotting
import matplotlib.pyplot as plt

def matrixY(movies, ratings, num_users):
    Y = np.zeros([len(movies), num_users])
    movies_dict = dict(zip(list(movies[:, 0]), range(len(movies))))
    for rating in ratings:
        Y[movies_dict[rating[1]], int(rating[0])] = float(rating[2])
    return Y

movies_file = "data/movies.csv"
ratings_file = "data/ratings.csv"

num_movies = 1000
num_users = 100
since_year = 2011
num_features = 10
lmbda = 0

m, r = preprocess(movies_file, ratings_file, num_movies, num_users, since_year)
Y = matrixY(m, r, num_users)

# Color plot of the user ratings
plt.imshow(Y)
plt.colorbar()
plt.ylabel("Movies")
plt.xlabel("Users")
plt.show()

Y_norm, Y_mean = normalize(Y)

# Set initial parameters (Theta, X)
Theta = np.random.uniform(low=-1, high=1, size=(num_users, num_features))
X = np.random.uniform(low=-1, high=1, size=(num_movies, num_features))
initial_params = np.concatenate((X.flatten(), Theta.flatten()))

J, grad = cofi_costfn.costfn(X, Theta, Y, num_users, num_movies, num_features, lmbda)
print("Cost: ", J)

# scipy.optimize.fmincg()
import numpy as np
from preprocess import preprocess
from normalize_ratings import normalize
from cofi_costfn import costfn, gradfn
from scipy import optimize
from feature_extraction import matrixY
import matplotlib.pyplot as plt

# Enter data set source path for movies and ratings
movies_file = "data/movies.csv"
ratings_file = "data/ratings.csv"

# Enter training data set size and learning parameters
num_movies = 100
num_users = 300
since_year = 2010
num_features = 10
lmbda = 0

m, r = preprocess(movies_file, ratings_file, num_movies, num_users, since_year)
# If number of movies found with the since condition is lesser than required, update the count
num_movies = m.shape[0]
Y = matrixY(m, r, num_users)

# Color plot of the user ratingss
plt.imshow(Y)
plt.colorbar()
plt.ylabel("Movies")
plt.xlabel("Users")
plt.show()

Y_mean, Y_norm  = normalize(Y)
Y = Y_norm

# Set initial parameters (Theta, X)
Theta = np.random.uniform(low=-1, high=1, size=(num_users, num_features))
X = np.random.uniform(low=-1, high=1, size=(num_movies, num_features))
initial_params = np.concatenate((X.flatten(), Theta.flatten()))

J = costfn(initial_params, Y, num_users, num_movies, num_features)
grad = gradfn(initial_params, Y, num_users, num_movies, num_features)

# Pre optimization cost
print("Cost: ", J)

args = (Y, num_users, num_movies, num_features)

# Minimizer options
opts = {'maxiter' : 100, 'disp' : True}

print "Training...."
# Nelder-Mead method -> Doesn't require a gradient function for minimization
optimized_params = optimize.minimize(costfn, initial_params,args=args, method="Nelder-Mead", options=opts)
print optimized_params

X = np.array(optimized_params.x[:num_movies * num_features]).reshape(num_movies, num_features)
Theta = np.array(optimized_params.x[X.size:]).reshape(num_users, num_features)

initial_params = np.concatenate((X.flatten(), Theta.flatten()))
J = costfn(initial_params, Y, num_users, num_movies, num_features)
# Post optimization cost
print("Cost: ", J)
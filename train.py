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
num_movies = 500
num_users = 4000
since_year = 2010
num_features = 10
lmbda = 0

m, r = preprocess(movies_file, ratings_file, num_movies, num_users, since_year)
# If number of movies found with the since condition is lesser than required, update the count
num_movies = m.shape[0]
Y = matrixY(m, r, num_users)

movies_dict = dict(zip(list(m[:, 0]), range(len(m))))
movie_titles = [x[1] for x in m]
# for i in zip(range(len(m)), movie_titles):
#     print i

# Enter ratings for a subset of movies of your choice
Y_custom_ratings = {9 : 5, 12: 5, 23: 5, 87: 5, 94: 2, 101: 5, 156: 5}
Y_choice = np.array([0] * len(m))
for k in Y_custom_ratings.keys():
    Y_choice[k] = Y_custom_ratings[k]

Y = np.append(Y, Y_choice.reshape(500, 1), 1)
num_users = Y.shape[1]

# Color plot of the user ratings
# plt.imshow(Y)
# plt.colorbar()
# plt.ylabel("Movies")
# plt.xlabel("Users")
# plt.show()

Y_mean, Y_norm = normalize(Y)

# Set initial parameters (Theta, X)
Theta = np.random.uniform(low=-1, high=1, size=(num_users, num_features))
X = np.random.uniform(low=-1, high=1, size=(num_movies, num_features))

initial_params = np.concatenate((X.flatten(), Theta.flatten()))
J = costfn(initial_params, Y, num_users, num_movies, num_features)
grad = gradfn(initial_params, Y, num_users, num_movies, num_features)

args = (Y, num_users, num_movies, num_features)

# Minimizer options
opts = {'maxiter' : 100, 'disp' : True}

print "Training...."
# Check gradient function implementation correctness
err = optimize.check_grad(costfn, gradfn, initial_params, Y, num_users, num_movies, num_features)
print "Gradient error: ", err
optimized_params = optimize.fmin_cg(costfn, initial_params, gradfn, args)
# Nelder-Mead method -> Doesn't require a gradient function for minimization
# optimized_params = optimize.minimize(costfn, initial_params,args=args, method="Nelder-Mead", options=opts)
# optimized_params = optimize.minimize(costfn, initial_params, method='BFGS', jac=gradfn, options=opts, args=args)

X = np.array(optimized_params[:num_movies * num_features]).reshape(num_movies, num_features)
Theta = np.array(optimized_params[X.size:]).reshape(num_users, num_features)

initial_params = np.concatenate((X.flatten(), Theta.flatten()))
J = costfn(initial_params, Y, num_users, num_movies, num_features)
# Post optimization cost
print("Cost: ", J)

# Predictions
p = np.mat(X) * np.mat(Theta).transpose()
predictions = p

n = 10 # n most recommended for me
print("Top " + str(n) + " recommendations")

# Get predictions for you for other movies
my_ratings = predictions[:, 0]
my_ratings = list(np.array(my_ratings))
s_ratings = [x[0] for x in my_ratings]
my_ratings.sort()

i = 1
j = 0
while j < n:
    if s_ratings.index(my_ratings[len(my_ratings) - i]) not in Y_custom_ratings.keys():
        print(m[s_ratings.index(my_ratings[len(my_ratings) - i])])
        j += 1
    i += 1
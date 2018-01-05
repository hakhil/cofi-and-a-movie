import numpy as np

def matrixY(movies, ratings, num_users):
    Y = np.zeros([len(movies), num_users])
    movies_dict = dict(zip(list(movies[:, 0]), range(len(movies))))
    for rating in ratings:
        Y[movies_dict[rating[1]], int(rating[0])] = float(rating[2])
    return Y

import csv
import re
import numpy as np

def movie_data(movies_file, num_movies, since_year):
    movies = []
    with open(movies_file) as csvfile:
        n = 0
        movie_reader = csv.reader(csvfile, delimiter=',')
        for row in movie_reader:
            i, name, genre = row
            year = re.findall("(\d+)", name)
            if len(year) == 0:
                pass
            else:
                if int(year[-1]) >= since_year:
                    movies.append([i, name, year[-1], genre])
                    n += 1
                    if n == num_movies:
                        break
    return np.array(movies)

def ratings_data(ratings_file, num_users, movie_ids):
    ratings = []
    with open(ratings_file) as csvfile:
        ratings_reader = csv.reader(csvfile, delimiter=',')
        c = 0
        for row in ratings_reader:
            i, movie_i, rating, ts = row
            if int(i) < num_users:
                if movie_i in movie_ids:
                    ratings.append([i, movie_i, rating, ts])
            else:
                break
    return np.array(ratings)

def preprocess(movies_file, ratings_file, num_movies, num_users, since_year):
    movies = movie_data(movies_file, num_movies, since_year)
    movie_ids = movies[:, 0]
    ratings = ratings_data(ratings_file, num_users, movie_ids)

    return movies, ratings
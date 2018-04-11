import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy as np
import numpy.random as nprand
import math
import scipy
import time
import math
import sys
from scipy import io
from scipy import stats
from datetime import datetime

class recommender():
    def __init__(self, ubreg=0.0, mbreg=0.0, ureg=0.0, mreg=0.0, gbreg=0.0, datafile='movie_training_data/user_ratedmovies_train.dat'):
        self.movie_data = pandas.read_csv(datafile,'\t')
        print 'movie_data size: {}'.format(len(self.movie_data))
        # create a test/train split
        val_size = 85000

        all_inds = nprand.permutation( range(0,len(self.movie_data)) )
        val_inds = all_inds[0:val_size]
        train_inds = all_inds[val_size:len(self.movie_data)]

        val = self.movie_data.iloc[ val_inds ].as_matrix()
        self.val_data = val[:,1:] # get all but the id
        train = self.movie_data.iloc[ train_inds ].as_matrix()
        self.train_data = train[:,1:] # get all but the id

        users = self.movie_data['userID'].unique()
        self.userSize = len(users)
        # print 'Number of Users: {}'.format(userSize)
        movies = self.movie_data['movieID'].unique()
        self.movieSize = len(movies)
        # print 'Number of Movies: {}'.format(movieSize)
        self.id2user = {}
        for i, user in enumerate(users):
            self.id2user[user] = i
        self.id2movie = {}
        for i, movie in enumerate(movies):
            self.id2movie[movie] = i

        self.ubreg = ubreg
        self.mbreg = mbreg
        self.ureg = ureg
        self.mreg = mreg
        self.gbreg = gbreg
        # global_bias is the average rating
        self.global_bias = np.mean(self.train_data[:,2])

        self.ufeat = None
        self.mfeat = None
        self.ubias = None
        self.mbias = None

    def reset(self, features):
        self.ufeat = nprand.normal(scale=1./features, size=(self.userSize, features))
        self.mfeat = nprand.normal(scale=1./features, size=(self.movieSize, features))
        self.ubias = np.zeros(self.userSize)
        self.mbias = np.zeros(self.movieSize)

    def predict(self, v1, v2, v3=None, v4=None):
        retval = np.dot(v1, v2) + self.global_bias
        if v3 is not None:
            retval += v3
        if v4 is not None:
            retval += v4
        return retval

    def train(self, features=10, lr=1e-3, iterations=100000):
        start_time = time.time()
        for it in range(iterations):
            rum = self.train_data[nprand.randint(0,self.train_data.shape[0])]
            uind = self.id2user[rum[0]]
            mind = self.id2movie[rum[1]]
            loss = (rum[2] - self.predict(self.ufeat[uind], self.mfeat[mind], self.ubias[uind], self.mbias[mind]))

            # self.global_bias += lr * (loss - self.global_bias * self.gbreg)
            self.ubias[uind] += lr * (loss - self.ubias[uind] * self.ubreg)
            self.mbias[mind] += lr * (loss - self.mbias[mind] * self.mbreg)
            self.ufeat[uind] = lr * (loss * self.mfeat[mind] - self.ureg * self.ufeat[uind])
            self.mfeat[mind] = lr * (loss * self.ufeat[uind] - self.mreg * self.mfeat[mind])

            # the calculations from the slides
            # self.ubias[uind] -= lr * (loss + self.ubias[uind] * self.ubreg)
            # self.mbias[mind] -= lr * (loss + self.mbias[mind] * self.mbreg)
            # self.ufeat[uind] -= lr * (loss * self.mfeat[mind] + self.ureg * self.ufeat[uind])
            # self.mfeat[mind] -= lr * (loss * self.ufeat[uind] + self.mreg * self.mfeat[mind])

            # lr *= 0.999
            # lr -= 0.001
        print( "Time to train: {}".format(time.time() - start_time) )

    def predict_set(self, dataset):
        predictions = np.zeros(dataset.shape[0])
        start_time = time.time()
        for it in range(dataset.shape[0]):
            rum = dataset[it]
            uind = self.id2user[rum[0]]
            mind = self.id2movie[rum[1]]
            predictions[it] += self.predict(self.ufeat[uind], self.mfeat[mind], self.ubias[uind], self.mbias[mind])
        print( "Time to evaluate: {}".format(time.time() - start_time) )
        return predictions

    def get_rmse(self, testset='validation'):
        if 'train' in testset:
            dataset = self.train_data
        elif 'val' in testset:
            dataset = self.val_data

        predictions = self.predict_set(dataset)
        rmse = np.sqrt(np.mean(np.power(dataset[:,2] - predictions, 2)))
        return rmse

    def explicit_train(self, features=10, lr=1e-4, iterations=100000, divisions=10):
        print 'Training'
        if self.ufeat is None:
            self.reset(features)
        print_every = iterations / divisions
        rmses = np.zeros(divisions+1)
        rmses[0] += self.get_rmse()
        print( "Iter {} RMSE: {}".format(0, rmses[0]) )
        for i in range(divisions):
            self.train(lr=lr, iterations=print_every)
            rmses[i+1] += self.get_rmse()
            print( "Iter {} RMSE: {}".format(i+1, rmses[i+1]) )
        return rmses

if __name__ == "__main__":
    lr = 1e-4
    iterations = 10000000 # 10,000,000
    features = 40
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    if len(sys.argv) > 2:
        lr = float(sys.argv[2])
    if len(sys.argv) > 3:
        features = int(sys.argv[3])
    system = recommender(0.01, 0.01, 0.01, 0.01, 0.01)
    system.explicit_train(features=features, lr=lr, iterations=iterations)

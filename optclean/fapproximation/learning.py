"""
This module defines the basic functionality for a data cleaning policy
"""
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree

from optclean.dcpolicy import *
from optclean.actions import *

import copy

import tensorflow as tf

class CleaningLearner(object):

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, dataset, policy):
        self.dataset = dataset
        self.policy = policy

        self.dimensions = self.policy._row2featureVector(dataset.df.iloc[0,:]).shape[0]

        self.X = tf.placeholder("float", [None, self.dimensions])
        self.Y = tf.placeholder("float", [None, self.dimensions])

        w1 = 0.01*tf.Variable(tf.random_normal([self.dimensions, 32]))
        b1 = 0.01*tf.Variable(tf.random_normal([32]))
        h1 = tf.nn.relu(tf.add(tf.matmul(self.X, w1), b1))

        w2 = 0.01*tf.Variable(tf.random_normal([32, 32]))
        b2 = 0.01*tf.Variable(tf.random_normal([32]))
        h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))

        w3 = 0.01*tf.Variable(tf.random_normal([32, self.dimensions]))
        b3 = 0.01*tf.Variable(tf.random_normal([self.dimensions]))
        self.out = self.X + tf.add(tf.matmul(h2, w3), b3)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(self.Y - self.out))

        self.sess = tf.Session()

    def sampleEdits(self):
        self.sampleDataset, self.edits = self.policy.run() 
        X = np.zeros( (len(self.edits), self.dimensions))
        Y = np.zeros( (len(self.edits), self.dimensions))
        for i, e in enumerate(self.edits):
            X[i,:] = e[3].reshape((1,-1))
            Y[i,:] = e[4].reshape((1,-1))

        return X,Y


    def train(self, learning_rate=1e-2, iters=10000, batch_frac=1):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        X, Y = self.sampleEdits()
        nrows, _ = X.shape

        init = tf.global_variables_initializer()

        self.sess.run(init)

        for epoch in range(iters):

            batch = np.random.choice(np.arange(0,nrows),size=int(batch_frac*nrows))

            self.sess.run(optimizer, {self.X: X[batch,:], self.Y: Y[batch,:]})

            if epoch % 100 == 0:
                print("Iteration Loss:", epoch, self.sess.run(self.loss, {self.X: X, self.Y: Y}))

    def apply(self, recursion_depth=10):

        def rapply(policy, attr, row):
            features = policy._row2featureVector(row)

            updateFeatures = self.sess.run(self.out, {self.X: features.reshape((1,-1))}).reshape((-1,))

            return policy._featureVector2attr(updateFeatures, attr)

        fnlist = []
        for t in self.policy.types:
            fnlist.append(lambda row, attr=t: rapply(self.policy, attr, row))

        return self.dataset.iterate(fnlist,[t for t in self.policy.types], max_iters=recursion_depth)










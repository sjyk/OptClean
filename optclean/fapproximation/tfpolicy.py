"""
This module defines the basic functionality for a data cleaning policy
"""
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree

from dcpolicy import *

import copy

import tensorflow as tf

class MultiLayerPerceptronPolicy(Policy):

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, dataset, types, config={}):
        super(MultiLayerPerceptronPolicy, self).__init__(dataset, types, config)

        d = np.squeeze(self.shape)
        self.fin = tf.placeholder("float", [None, d])

        h1 = tf.Variable(0*tf.random_normal([d, 32]))
        b1 = tf.Variable(0*tf.random_normal([32]))
        net = tf.add(tf.matmul(self.fin, h1), b1)
        net = tf.nn.relu(net)

        h2 = tf.Variable(0*tf.random_normal([32, d]))
        b2 = tf.Variable(0*tf.random_normal([d]))

        self.out = tf.add(tf.add(tf.matmul(net, h2), b2), self.fin) 

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def _getParams(self):
        return self.sess.run(tf.trainable_variables())


    def _setParams(self, weights):
        for i,v in enumerate(tf.trainable_variables()):
            self.sess.run(tf.assign(v, weights[i]))


    def _eval(self, f):
        return self.sess.run(self.out, {self.fin: f.reshape((1,-1))}).reshape((-1))







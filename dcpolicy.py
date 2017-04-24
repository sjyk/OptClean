"""
This module defines the basic functionality for a data cleaning policy
"""
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree

from dataset import *

import copy

class Policy:

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, dataset, types, config={}):
        self.dataset = dataset
        self.types = types
        self.featurizers = {}

        for t in types:
            if types[t] == 'num':
                self.featurizers[t] = NumericalFeatureSpace(dataset.df, t)
            elif types[t] == 'cat':
                if t in config:
                    self.featurizers[t] = CategoricalFeatureSpace(dataset.df, t, **config[t])
                else:
                    self.featurizers[t] = CategoricalFeatureSpace(dataset.df, t)

        #initializes the data structure
        self._row2featureVector(dataset.df.iloc[0,:])

    """
    This function converts a row into a feature vector
    """
    def _row2featureVector(self, row):

        vlist = []

        self.attrIndex = {}
        start = 0

        for t in self.types:
            print(t, row[t])
            v = np.array(self.featurizers[t].val2feature(row[t]))
            
            if self.types[t] == 'cat':
                dims = np.squeeze(v.shape)
            else:
                dims = 1

            self.attrIndex[t] = (start, start+dims)
            start = start + dims
            vlist.append(v)

        return np.hstack(tuple(vlist))


    """
    This function converts a row into a val for the desired attribute
    """
    def _featureVector2attr(self, f, attr):
        vindices = self.attrIndex[attr]
        return self.featurizers[attr].feature2val(f[vindices[0]:vindices[1]])


    def _eval(self, f):
        return f + np.random.randn(f.shape[0])


    def run(self, config={}):
        attrlist = [t for t in self.types]
        
        fnlist = []


        def rapply(policy, attr, row):
            features = policy._row2featureVector(row)
            out = policy._eval(features)
            print("__",attr,policy._featureVector2attr(out, attr))
            return policy._featureVector2attr(out, attr)

        fnlist.append(lambda row: rapply(self, 'AAA', row))
        fnlist.append(lambda row: rapply(self, 'BBB', row))


        dataset, result = self.dataset.iterate(fnlist,attrlist,max_iters=1)

        print(dataset.df)






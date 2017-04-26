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

class Policy(object):

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
        tmp = self._row2featureVector(dataset.df.iloc[0,:])
        self.shape = tmp.shape

    """
    This function converts a row into a feature vector
    """
    def _row2featureVector(self, row):

        vlist = []

        self.attrIndex = {}
        start = 0

        for t in self.types:
            #print(t, row[t])
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


    def _sampleBatch(self, f, step, batch_size):
        params = []

        for i in range(batch_size):
            mask = np.zeros(f.shape)
            maskind = self.attrIndex[np.random.choice(list(self.attrIndex.keys()))]
            mask[maskind[0]:maskind[1]] = 1

            params.append(f + step*mask*np.random.randn(np.squeeze(f.shape)))

        params.append(f)

        return params

    def _searchBatch(self, i, step, batch_size):
        attrlist = [t for t in self.types]

        batch = self._sampleBatch(self._row2featureVector(self.dataset.df.iloc[i,:]), step, batch_size)

        def rapply(policy, attr, f, j, row):
            features = policy._row2featureVector(row)
            if j == row.name:
                return policy._featureVector2attr(f, attr)
            return policy._featureVector2attr(features, attr)

        batch_results = []

        for b in batch:
            
            fnlist = []
            for t in self.types:
                fnlist.append(lambda row, attr=t: rapply(self, attr, b, i, row))

            dataset, result = self.dataset.iterate(fnlist,attrlist,max_iters=2)
            batch_results.append((result[-1]['score'],dataset))

        return sorted(batch_results, key=lambda x: x[0])


    def run(self, config={}):
        rows, cols = self.dataset.df.shape

        for t in range(10):
            i = np.random.choice(np.arange(0,rows))
            b = self._searchBatch(i, 10.0/(t+1), 10)
            self.dataset = b[-1][1]
            print("Iteration", t, b[-1][0])

        return self.dataset






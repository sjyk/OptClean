"""
This module defines the basic functionality for a data cleaning policy
"""
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree

from dataset import *
from actions import *

class Policy(object):

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, 
                 dataset, 
                 types, 
                 stepsize=100, 
                 iterations=100, 
                 batchsize=10, 
                 rollout=2):

        self.dataset = dataset
        self.types = types
        self.featurizers = {}

        for t in types:
            if types[t] == 'num':
                self.featurizers[t] = NumericalFeatureSpace(dataset.df, t)
            elif types[t] == 'cat':
                self.featurizers[t] = CategoricalFeatureSpace(dataset.df, t)

        #initializes the data structure
        tmp = self._row2featureVector(dataset.df.iloc[0,:])
        self.shape = tmp.shape


        self.aconfig = {}
        self.aconfig['step'] = stepsize
        self.aconfig['iterations'] = iterations
        self.aconfig['batch'] = batchsize
        self.aconfig['rollout'] = rollout

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
        rows, cols = self.dataset.df.shape

        for i in range(batch_size):
            typed_keys = [k for k in self.attrIndex.keys() if k in self.types]
            attr = np.random.choice(typed_keys)
            dtype = self.types[attr] 

            action = np.random.choice(getPossibleActions(dtype))

            if action['params'] == 'value':
                vparam = np.random.choice(self.dataset.df[attr].values)
                #print(vparam)
            else:
                vparam = None

            #print(i, attr, vparam, action['fn'](attr, f, vparam))

            params.append(action['fn'](attr, f, vparam))

        params.append(f.copy(deep=True))

        #print('---',f,params)

        return params

    def _searchBatch(self, i, step, batch_size):
        attrlist = [t for t in self.types]

        batch = self._sampleBatch(self.dataset.df.iloc[i,:], step, batch_size)
        #print(batch)


        def rapply(policy, attr, f, j, row):
            features = policy._row2featureVector(row)
            updateFeatures = policy._row2featureVector(f)

            if j == row.name:
                return policy._featureVector2attr(updateFeatures, attr)

            return policy._featureVector2attr(features, attr)

        batch_results = []

        for b in batch:

            ec = self.editCost(self.dataset.provenance.iloc[i,:], b)
            
            fnlist = []
            for t in self.types:
                fnlist.append(lambda row, update=b, attr=t: rapply(self, attr, update, i, row))

            dataset, result = self.dataset.iterate(fnlist,attrlist,max_iters=self.aconfig['rollout'])
            batch_results.append((result[-1]['score'], ec , dataset))

        return sorted(batch_results, key=lambda x: x[0:1])


    def run(self, config={}):
        rows, cols = self.dataset.df.shape

        for t in range(self.aconfig['iterations']):
            i = np.random.choice(np.arange(0,rows))
            b = self._searchBatch(i, self.aconfig['step'], self.aconfig['batch'])
            self.dataset = b[-1][2]
            print("Iteration", i, t, b[-1][0])

        return self.dataset


    def editCost(self, row, update):
        #print(row, update)
        return -np.linalg.norm(self._row2featureVector(row)-self._row2featureVector(update))






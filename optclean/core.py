"""
This module implements the core elements of the optclean packaged
"""

import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree
import distance 
from sklearn import tree
from constraints import *


class Dataset:

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, df, types, provenance=-1):
        self.df = df

        try:
            int(provenance)
            self.provenance = pd.DataFrame.copy(df)
        except:
            self.provenance = provenance


        self.types = types
        self.featurizers = {}

        for t in types:

            if types[t] == 'num':
                self.featurizers[t] = NumericalFeatureSpace(df, t)
            elif types[t] == 'cat':
                self.featurizers[t] = CategoricalFeatureSpace(df, t)
            elif types[t] == 'string':
                self.featurizers[t] = StringFeatureSpace(df, t)

        #print(self.featurizers)

        #initializes the data structure
        tmp = self._row2featureVector(self.df.iloc[0,:])
        self.shape = tmp.shape



    """
    Internal function that creates a new dataset
    with fn mapped over all records
    """
    def _map(self, fn, attr):
        newDf = pd.DataFrame.copy(self.df)
        rows, cols = self.df.shape

        j = newDf.columns.get_loc(attr)
        for i in range(rows):
            newDf.iloc[i,j] = fn(newDf.iloc[i,:])
            #print("++",j,newDf.iloc[i,j], fn(newDf.iloc[i,:]))

        return Dataset(newDf, 
                       self.qfnList, 
                       self.provenance)

    def _sampleRow(self):
        newDf = pd.DataFrame.copy(self.df)
        rows, cols = self.df.shape
        i = np.random.choice(np.arange(0,rows))
        return newDf.iloc[i,:]


    """
    Executes fixes over all attributes in a random
    order
    """
    def _allmap(self, fnList,  attrList):

        dataset = self
        for i, f in enumerate(fnList):
            dataset = dataset._map(f,attrList[i])

        return dataset
        

    """
    Evaluates the quality metric
    """
    def getPredicates(self, qfn):

        from sklearn.tree import _tree

        
        #TODO fix a parameter
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1)

        N = self.df.shape[0]
        
        X = np.zeros((N,self.shape[0]))

        for i in range(N):
            X[i,:] = self._row2featureVector(self.df.iloc[i,:])

        labels = np.sign(qfn(self.df))
        
        if np.sum(labels) == 0 or np.sum(labels) == N:
            #print("hi")
            return [lambda row: False]

        clf = clf.fit(X, np.sign(qfn(self.df)))

        paths = self.get_lineage(clf)


        def path2predicate(row, path, dataset=self):
            satisfies = True

            features = dataset._row2featureVector(row)

            for node in path:

                if node[1] == 'r':

                    if features[node[3]] < node[2]:

                        satisfies = False

                else:

                    if features[node[3]] >= node[2]:

                        satisfies = False

            return satisfies

        print('--',paths)

        return [lambda row, path=p, dataset=self: path2predicate(row, path, dataset) for p in paths]

    def get_lineage(self, tree):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value
        features  = [i for i in tree.tree_.feature]

         # get ids of child nodes
        idx = np.argwhere(left == -1)[:,0]     

        def recurse(left, right, child, lineage=None):          
              if lineage is None:
                   lineage = [child]
              if child in left:
                   parent = np.where(left == child)[0].item()
                   split = 'l'
              else:
                   parent = np.where(right == child)[0].item()
                   split = 'r'

              lineage.append((parent, split, threshold[parent], features[parent]))

              if parent == 0:
                   lineage.reverse()
                   return lineage
              else:
                   return recurse(left, right, parent, lineage)


        paths = []
        running = []
        for child in idx:
              for node in recurse(left, right, child):
                    try:
                        n = int(node)

                        if np.argmax(value[node, :]) == 1:
                            paths.append(running)

                        #
                        #if np.argmax(value[node, :]):
                        #    paths.append(running)

                        running = []
                    except:
                        running.append(node)

        print(paths)

        return paths




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



"""
This class defines a categorical feature space 
"""
class StringFeatureSpace:

    """
    Initializes with hyperparams
    """
    def __init__(self, df, attr, simfn=distance.levenshtein, maxDims=2, var=1, ballTreeLeaf=2):

        self.df = df
        self.attr = attr
        self.simfn = simfn

        self.maxDims = maxDims
        self.var = var
        self.ballTreeLeaf = ballTreeLeaf

        self.values =  set(self.df[self.attr].values)

        self.tree, self.findex, self.iindex = self._calculateEmbedding()


    """
    Embeds the categorical values in a vector space
    """
    def _calculateEmbedding(self):
        values = self.values
        kernel = np.zeros((len(values), len(values)))

        for i, v in enumerate(values):
            for j, w in enumerate(values):
                if i != j:
                    kernel[i,j] = np.exp(-self.simfn(v,w)/self.var)

        embed = spectral_embedding(kernel, n_components=self.maxDims)
        tree = BallTree(embed, leaf_size=self.ballTreeLeaf)

        return tree, {v: embed[i,:] for i, v in enumerate(values)}, {i: v for i, v in enumerate(values)}


    """
    Calculates the value of a feature vector
    """
    def feature2val(self, f):
        _ , result = self.tree.query(f.reshape(1, -1), k=1)
        return self.iindex[result[0][0]]


    """
    Calculates the features of a value
    """
    def val2feature(self, val):
        #print(val)
        #snap to nearest value if not there
        if val in self.values:
            return self.findex[val]
        else:
            
            minDistance = np.inf
            minIndex = -1

            for i, v in enumerate(self.values):
                if self.simfn(val, v) < minDistance:
                    minDistance = self.simfn(val, v)
                    minIndex = i

            return self.findex[self.iindex[minIndex]]


"""
This class defines a categorical feature space 
"""
class CategoricalFeatureSpace:

    """
    Initializes with hyperparams
    """
    def __init__(self, df, attr):

        self.df = df
        self.attr = attr
        self.values =  set(self.df[self.attr].values)
        self.vindex = {i:v for i,v in enumerate(self.values)}
        self.iindex = {v:i for i,v in enumerate(self.values)}

        print(self.vindex)


    """
    Calculates the value of a feature vector
    """
    def feature2val(self, f):
        index = np.argmax(f)
        return self.vindex[index]


    """
    Calculates the features of a value
    """
    def val2feature(self, val):
        #print(val)
        f = np.zeros((len(self.values),))
        if val in self.values:
            f[self.iindex[val]] = 1.0
        
        return f


"""
This class defines a numerical feature space 
"""
class NumericalFeatureSpace:
    """
    Initializes with hyperparams
    """
    def __init__(self, df, attr):

        self.df = df
        self.attr = attr
    
    def feature2val(self, f):
        return np.squeeze(f)

    def val2feature(self, val):
        return val


    




 
"""
This module defines the basic functionality for a dataset
"""
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree
import distance 

class Dataset:

    """
    A dataset takes a data frame as input and a list of
    quality functions
    """
    def __init__(self, df, qfnList = [], provenance=-1):
        self.df = df
        self.qfnList = qfnList

        try:
            int(provenance)
            self.provenance = pd.DataFrame.copy(df)
        except:
            self.provenance = provenance


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
    def _evalQ(self):
        baseline = [qfn(self.provenance) for qfn in self.qfnList]
        scores = [qfn(self.df) for qfn in self.qfnList]
        merged = self.df.merge(self.provenance, indicator=True, how='outer')
        ro = merged[merged['_merge'] == 'right_only'].shape[0]
        lo = merged[merged['_merge'] == 'left_only'].shape[0]
        edit = (ro + lo)/2
        
        return { 'baseline': np.sum(baseline),
                 'score': np.sum(scores),
                 'edit': edit,
                 'rawBaseline': baseline,
                 'rawScores': scores
               }

    """
    Adds a quality function to the dataframe
    """
    def addQualityMetric(self, qfn):
        self.qfnList.append(qfn)


    """
    Fixed point iterate-applies the chase until convergence
    """
    def iterate(self, fnList, attrList, max_iters=10):
        
        dataset = self

        results = []

        for it in range(max_iters):
            results.append(dataset._evalQ())
            dataset = dataset._allmap(fnList, attrList)

        return dataset, results





"""
This class defines a categorical feature space 
"""
class CategoricalFeatureSpace:

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



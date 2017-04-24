
import pandas as pd
import numpy as np
import random
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import BallTree
import distance 
from dcpolicy import Policy
from dataset import *


df = pd.DataFrame({'AAA' : ['USA','United States','China','Canada'], 'BBB' : [10,20,30,40],
                   'CCC' : [100,50,-30,-50]})
print(df)
d = Dataset(df)

c = CategoricalFeatureSpace(df, 'AAA', distance.levenshtein)
print(c.feature2val([np.array([2,2])]))
print(c.val2feature("US"))


n = NumericalFeatureSpace(df, 'BBB')
print(n.val2feature(5))
print(n.feature2val(5))

p = Policy(d, {'AAA': 'cat', 'BBB': 'num'})
print(p._row2featureVector(df.iloc[0,:]))
print(p._featureVector2attr(p._row2featureVector(df.iloc[0,:]), 'BBB'))
p.run()
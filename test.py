
import pandas as pd
from optclean import *

df = pd.DataFrame({'AAA' : ['USA','United States','China','Canada'], 'BBB' : [10,15,30,40],
                   'CCC' : [100,50,-30,-50]})

print(df)

d = Dataset(df)

#qfn = lambda a: -len(set(a['AAA'].values))

#d.addQualityMetric(qfn)

#qfn = lambda a: -np.sum(np.abs(np.array(a['BBB'].values)))


qfn = lambda a: -np.abs(np.sum(a['BBB'].values) - 100)


d.addQualityMetric(qfn)

"""
c = CategoricalFeatureSpace(df, 'AAA', distance.levenshtein)
print(c.feature2val([np.array([2,2])]))
print(c.val2feature("US"))


n = NumericalFeatureSpace(df, 'BBB')
print(n.val2feature(5))
print(n.feature2val(5))
"""

p = Policy(d, {'AAA': 'cat', 'BBB': 'num'})
print(p.run().df)
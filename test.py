
import pandas as pd
from optclean import *

## Example 1
"""
df = pd.DataFrame({'vals': [1,2,3,4,5,100,9,4,3]})
d = Dataset(df)

qfn = lambda a: -np.sum(np.abs(np.array(a['vals'].values) - \
                         np.median(np.array(a['vals'].values))) > 10)
d.addQualityMetric(qfn)

p = Policy(d, {'vals': 'num'}, stepsize=100, batchsize=100, iterations=25)
print(p.run().df)
"""



## Example 2
df = pd.DataFrame({'Name' : ['United States','United States','China','Canada','Brazil'], 'Code' : ['US','USA','CN','CA','BA']})
d = Dataset(df)

def qfn(a):
    violations = 0

    for n in a['Name'].values:
        if len(set(a.where(a['Name']==n)['Code'].dropna().values)) > 1:
            violations = violations + 1

    return -violations

d.addQualityMetric(qfn)

p = Policy(d, {'Name': 'cat', 'Code': 'cat'}, stepsize=10, batchsize=10, iterations=5)
print(p.run().df)



"""
## Example 3
df = pd.DataFrame({'Name' : ['United States','United States','China','Canada','Brazil'], 'Code' : ['US','USA','CN','CA','BA']})
d = Dataset(df)

qfn = lambda a: -len(set([len(n) for n in a['Code'].values]))

d.addQualityMetric(qfn)

p = Policy(d, {'Name': 'cat', 'Code': 'cat'}, stepsize=10, batchsize=10, iterations=5)
print(p.run().df)
"""







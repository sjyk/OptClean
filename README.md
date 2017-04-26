# OptClean
Optimization-based synthesis for data cleaning.

## What Does OptClean Do
OptClean uses a randomized search algorithm over possible modifications to a Pandas data frame to best 
improve a user defined quality metric.

## Examples

### Outlier Removal
First, import the package:
```
import pandas as pd
from optclean import *
```

Then, let's create a basic numerical dataset:
```
df = pd.DataFrame({'vals': [1,2,3,4,5,100,9,4,3]})
d = Dataset(df)
```
There is one outlier at 100.

We can define a quality function that counts the number of elements greater than 10 from the mean:
```
qfn = lambda a: -np.sum(np.abs(np.array(a['vals'].values) - \
                         np.median(np.array(a['vals'].values))) > 10)
d.addQualityMetric(qfn)
```

Then, we run the optimization algorithm:
```
p = Policy(d, {'vals': 'num'})
print(p.run().df)

0  1.000000
1  2.000000
2  3.000000
3  4.000000
4  5.000000
5  5.437957
6  9.000000
7  4.000000
8  3.000000
```

### Functional Dependency
The same algorithm can also apply to FD violations (Name -> Code):
```
df = pd.DataFrame({'Name' : ['United States','United States','China','Canada'], 'Code' : ['US','USA','CN','CA']})
d = Dataset(df)
```

Define a quality function to count the FD violations:
```
def qfn(a):
    violations = 0

    for n in a['Name'].values:
        if len(a.where(a['Name']==n)['Code'].dropna().values) > 1:
            violations = violations + 1

    return -violations
```

```
p = Policy(d, {'Name': 'cat', 'Code': 'cat'}, stepsize=10, batchsize=10, iterations=20)
print(p.run().df)
  Code           Name
0   US  United States
1   US  United States
2   CN          China
3   CA         Canada
4   BA         Brazil
```

### Exotic Quality Metrics
Same dataset as above distinct string lengths in codes:
```
qfn = lambda a: -len(set([len(n) for n in a['Code'].values]))
```

Replaces USA with CN
```
p = Policy(d, {'Name': 'cat', 'Code': 'cat'}, stepsize=10, batchsize=10, iterations=20)
print(p.run().df)
  Code           Name
0   US  United States
1   CN  United States
2   CN          China
3   CA         Canada
4   BA         Brazil
```


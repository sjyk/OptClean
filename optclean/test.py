import pandas as pd
from search import *
from ops import *


"""
Using OptClean for Functional Dependencies (e.g., NADEEF VLDB 2012)
"""
data = [{'a': 'New York', 'b': 'A'}, 
         {'a': 'New York', 'b': 'NY'}, 
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Jose', 'b': 'SJ'},
         {'a': 'New York', 'b': 'NY'},
         {'a': 'San Francisco', 'b': 'SFO'},
         {'a': 'Berkeley', 'b': 'Bk'},
         {'a': 'Oakland', 'b': 'Oak'}]

df = pd.DataFrame(data)
fd = FD(["a"], ["b"]) #create an FD a->b
qualityFn = fd.qfn
operation = treeSearch(df, qualityFn, [Swap])
print(operation.run(df))



"""
Using OptClean for Multiple Functional Dependencies (e.g., Holistic Data Cleaning ICDE 2015)
"""
data2 = [{'a': 'New York', 'b': 'A'}, 
         {'a': 'New York', 'b': 'NY'}, 
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Jose', 'b': 'SJ'},
         {'a': 'New York', 'b': 'NY'},
         {'a': 'San Francisco', 'b': 'SFO'},
         {'a': 'Berkeley', 'b': 'Bk'},
         {'a': 'Oakland', 'b': 'Oak'}]

df = pd.DataFrame(data2)
fd = FD(["a"], ["b"]) #create an FD a->b
fd2 = FD(["b"], ["a"]) #create an FD b->a 

qualityFn = (fd * fd2).qfn #"one-to-one" constraint

operation = treeSearch(df, qualityFn, [Swap])
print(operation.run(df))



"""
Using OptClean for Domain Integrity Constraints
"""
data3 = [{'a': 'New York', 'b': 'A'}, 
         {'a': 'New York', 'b': 'NY'}, 
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Jose', 'b': 'SJ'},
         {'a': 'New York', 'b': 'NY'},
         {'a': 'San Francisco', 'b': 'SFO'},
         {'a': 'Berkeley', 'b': 'Bk'},
         {'a': 'Oakland', 'b': 'Oak'}]
df = pd.DataFrame(data3)

di = DictValue('a', ['New York', 'San Francisco', 'San Jose', 'Oakland', 'Berkeley'])
editCost = CellEdit(df)
qualityFn = (di + editCost).qfn #minimize editing of cells
operation = treeSearch(df, qualityFn, [Swap])



"""
Using OptClean for Schema Transformations (e.g., Potters Wheel VLDB 2001)
"""
data4 = [{'a': 'Miami-A'}, 
         {'a': 'New York-NY'}, 
         {'a': 'San Francisco-SF'},
         {'a': 'San Francisco-SF'},
         {'a': 'San Jose-SJ'},
         {'a': 'New York-NY'},
         {'a': 'San Francisco-SF'},
         {'a': 'Berkeley-Bk'},
         {'a': 'Oakland-Oak'}]

df = pd.DataFrame(data4)
shapeConstraint = Shape(9, 2)
predicate = Predicate(None, lambda x: x != None) #count null values
qualityFn = (shapeConstraint + predicate).qfn #minimize editing of cells
operation = treeSearch(df, qualityFn, [Fold, Split, Unfold, Divide, MergeNull])



"""
Using OptClean for Scorpion (VLDB 2013)
"""
data5 = [{'a': 'New York City', 'b': 1}, 
         {'a': 'New York', 'b': 2}, 
         {'a': 'San Francisco', 'b': 1},
         {'a': 'San Francisco', 'b': 1},
         {'a': 'San Jose', 'b': 1},
         {'a': 'New York', 'b': 500},
         {'a': 'San Francisco', 'b': 1},
         {'a': 'Berkeley', 'b': 1},
         {'a': 'Oakland', 'b': 1}]

df = pd.DataFrame(data4)
agg = Agg('sum','b', 'a', lambda x: x < 100)
editCost = CellEdit(df)
qualityFn = (agg + editCost).qfn
operation = treeSearch(df, qualityFn, [Delete])
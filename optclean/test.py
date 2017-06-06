import pandas as pd
from search import *

h2 = []

data2 = [{'a': 'New York', 'b': 'A'}, 
         {'a': 'New York', 'b': 'NY'}, 
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Jose', 'b': 'SJ'},
         {'a': 'New York', 'b': 'NY'},
         {'a': 'San Francisco', 'b': 'SFO'},
         {'a': 'Berkeley', 'b': 'Bk'},
         {'a': 'Oakland', 'b': 'Oak'}]

data3 = [{'a': 'New York City', 'b': 'NY'}, 
         {'a': 'New York', 'b': 'NY'}, 
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Francisco', 'b': 'SF'},
         {'a': 'San Jose', 'b': 'SJ'},
         {'a': 'New York', 'b': 'NY'},
         {'a': 'San Francisco', 'b': 'SFO'},
         {'a': 'Berkeley', 'b': 'Bk'},
         {'a': 'Oakland', 'b': 'Oak'}]

data = [{'a': 'Miami-A'}, 
         {'a': 'New York-NY'}, 
         {'a': 'San Francisco-SF'},
         {'a': 'San Francisco-SF'},
         {'a': 'San Jose-SJ'},
         {'a': 'New York-NY'},
         {'a': 'San Francisco-SF'},
         {'a': 'Berkeley-Bk'},
         {'a': 'Oakland-Oak'}]


"""
df = pd.DataFrame(data3)

f = FD(["a"], ["b"])

g = FD(["b"], ["a"])



i = Shape(9, 2)

operation = treeSearch(df, (f*g + h*2).qfn, [Swap])

print(operation.run(df))
"""

df = pd.DataFrame(data3)

f = DictValue('a', ['New York', 'San Francisco', 'San Jose', 'Oakland', 'Berkeley'])

g = FD(["a"], ["b"])

h = FD(["b"], ["a"])

i = CellEdit(df.copy(deep=True))



operation = treeSearch(df, (f + i + (h*g) ).qfn, [Swap])

print(operation.run(df))
print("______")
print((i).qfn(operation.run(df)))

#print((f+h).qfn(operation.run(df)))
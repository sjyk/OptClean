"""
This module generates feasible parameter settings, the settings 
are in a form of an ordered list
"""
from itertools import combinations, product
from ops import *
from constraints import *
from core import *
import copy


class ParameterSampler(object):

    def __init__(self, df, qfn, operationList, substrThresh=0.5, scopeLimit=3):
        self.df = df
        self.qfn = qfn
        self.substrThresh = substrThresh
        self.scopeLimit = scopeLimit
        self.operationList = operationList

        #TODO fix
        self.dataset = Dataset(df, {'a':'cat', 'b':'cat'})
        #self.dataset = Dataset(df, {'a':'cat'})


    def getParameterGrid(self):
        parameters = []
        paramset = [(op, sorted(op.paramDescriptor.values()), op.paramDescriptor.values()) for op in self.operationList]

        for op, p, orig in paramset:

            if p[0] == ParametrizedOperation.COLUMN:

                #remove one of the cols
                origParam = copy.copy(orig)
                orig.remove(p[0])
                colParams = []


                for col in self.columnSampler():

                    grid = []

                    for pv in orig:

                        grid.append(self.indexToFun(pv, col))

                    
                    #todo fix
                    augProduct = []
                    for p in product(*grid):
                        v = list(p)
                        v.insert(0, col)
                        augProduct.append(tuple(v))

                    colParams.extend(augProduct)

                parameters.append((op, colParams, origParam))

            else:
                
                grid = []

                for pv in orig:

                    grid.append(self.indexToFun(pv))


                parameters.append( (op, product(*grid), orig))

        #print(parameters)

        return parameters


    def getAllOperations(self):

        parameterGrid = self.getParameterGrid()
        operations = []

        for i , op in enumerate(self.operationList):
            args = {}

            #print(parameterGrid[i][1])
            
            for param in parameterGrid[i][1]:
                arg = {}
                for j, k in enumerate(op.paramDescriptor.keys()):
                    arg[k] = param[j]
                #print(arg)
                operations.append(op(**arg))

        return operations 


    def indexToFun(self, index, col=None):
        if index == ParametrizedOperation.COLUMN:
            return self.columnSampler()
        elif index == ParametrizedOperation.COLUMNS:
            return self.columnsSampler()
        elif index == ParametrizedOperation.VALUE:
            return self.valueSampler(col)
        elif index == ParametrizedOperation.SUBSTR:
            return self.substrSampler(col)
        elif index == ParametrizedOperation.PREDICATE:
            return self.predicateSampler(col)
        else:
            raise ValueError("Error in: " + index)


    def columnSampler(self):
        return self.df.columns.values.tolist()


    def columnsSampler(self):
        columns = self.columnSampler()
        result = []
        for i in range(1, min(len(columns), self.scopeLimit)):
            result.extend([list(a) for a in combinations(columns, i)])

        return result


    def valueSampler(self, col):
        #print("--",col, list(set(self.df[col].values)))
        return list(set(self.df[col].values))


    def substrSampler(self, col):
        chars = {}
        
        for v in self.df[col].values:
            for c in set(v):
                if c not in chars:
                    chars[c] = 0

                chars[c] += 1

        return [c for c in chars if (chars[c]+0.)/self.df.shape[0] > self.substrThresh]


    """
    Brute Force
    def predicateSampler(self, col):
        columns = self.columnSampler()
        columns.remove(col)
        projection = self.df[columns]
        tuples = set([tuple(x) for x in projection.to_records(index=False)])

        result_list = []
        for t in tuples:
            result_list.append(lambda s, p=t: (s[columns].values.tolist() == list(p)))

        return result_list
    """

    def predicateSampler(self, col):

        return self.dataset.getPredicates(self.qfn)



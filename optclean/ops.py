"""
This class defines the operations that we can search over.

Operations define a monoid
"""
from sets import Set


"""
Allows lazy composition of Op functions
"""
class Operation(object):

    def __init__(self, runfn):
        self.runfn = lambda df: runfn(df) 

    """
    This runs the operation
    """
    def run(self, df):
        df_copy = df.copy(deep=True)

        return self.runfn(df_copy)

    """
    Defines composable operations on a data frame
    """
    def __mul__(self, other):
        new_runfn = lambda df, a=self, b=other: a.runfn(b.runfn(df))
        new_op = Operation(new_runfn)
        return new_op

    """
    Easy to specify fixed point iteration
    """
    def __pow__(self, b):
        op = self

        for i in range(b):
            op *= self
        
        return op


"""
A parametrized operation is an operation that
takes parameters
"""
class ParametrizedOperation(Operation):

    COLUMN = 0
    VALUE = 1
    SUBSTR = 2
    PREDICATE = 3
    COLUMNS = 4

    def __init__(self, runfn, params):

        self.validateParams(params)
        super(ParametrizedOperation,self).__init__(runfn)



    def validateParams(self, params):

        try:
            self.paramDescriptor
        except:
            raise NotImplemented("Must define a parameter descriptor")

        for p in params:

            if p not in self.paramDescriptor:
                raise ValueError("Parameter " + str(p) + " not defined")

            if self.paramDescriptor[p] not in range(5):
                raise ValueError("Parameter " + str(p) + " has an invalid descriptor")



"""
Potter's wheel operations
"""

class Split(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN, 'delim': ParametrizedOperation.SUBSTR}

    def __init__(self, column, delim):

        def fn(df, column=column, delim=delim):

            args = {}

            #get the split length
            length = df[column].map(lambda x: len(x.split(delim))).max()

            def safeSplit(s, delim, index):
                splitArray = s.split(delim)
                if index >= len(splitArray):
                    return None
                else:
                    return splitArray[index]

            for l in range(length):
                args[column+str(l)] = df[column].map(lambda x, index=l: safeSplit(x, delim, l))

            return df.assign(**args)

        super(Split,self).__init__(fn, ['column', 'delim'])




class Drop(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN}

    def __init__(self, column):

        def fn(df, column=column):

            return df.drop(column, axis=1)

        super(Drop,self).__init__(fn, ['column'])



class Divide(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN, 
                                'substr': ParametrizedOperation.SUBSTR}

    def __init__(self, column, substr):

        def fn(df, 
               column=column, 
               predicate=lambda s: substr in s):

            args = {}

            def div(s, predicate, flip=False):

                if not flip:
                    indicator = predicate(s)
                else:
                    indicator = not predicate(s)

                if indicator:
                    return None
                else:
                    return s

            args[column+'0'] = df[column].map(lambda x, predicate=predicate: div(x, predicate, False))
            args[column+'1'] = df[column].map(lambda x, predicate=predicate: div(x, predicate, True))

            return df.assign(**args)

        super(Divide,self).__init__(fn, ['column', 'substr'])


class Fold(ParametrizedOperation):

    paramDescriptor = {'columnsKey': ParametrizedOperation.COLUMNS, 
                       'columnsTarget': ParametrizedOperation.COLUMNS}

    def __init__(self, columnsKey, columnsTarget):

        def fn(df, c1=columnsKey, c2=columnsTarget):
            
            if df[c1].drop_duplicates().shape[0] != df.shape[0]:
                raise ValueError("Not a Key")

            df1 = df

            args = {}
            for a in df.columns.values:

                if a in c2:
                    args[a] = df1[a].map(lambda x: None)

            df1 = df1.assign(**args)


            args = {}
            df2 = df.copy(deep=True)

            for a in df.columns.values:

                if a not in c1 and a not in c2:
                    args[a] = df2[a].map(lambda x: None)

            df2 = df2.assign(**args)

            return df1.append(df2)

        super(Fold,self).__init__(fn, ['columnsKey', 'columnsTarget'])




class MergeEmpty(ParametrizedOperation):

    paramDescriptor = {'column1': ParametrizedOperation.COLUMN, 
                       'column2': ParametrizedOperation.COLUMN}

    def __init__(self, column1, column2):

        def fn(df, c1=column1, c2=column2):

            args = {}
            
            def switch(x, c1, c2):
                if x[c1] == None and not x[c2] == None:
                    return x[c2]
                elif x[c2] == None and not x[c1] == None:
                    return x[c1]
                elif x[c2] == None and x[c1] == None:
                    return None
                else:
                    raise ValueError("Two values")

            def argswitch(x, c1, c2):
                if x[c1] == None and not x[c2] == None:
                    return c2
                elif x[c2] == None and not x[c1] == None:
                    return c1
                elif x[c2] == None and x[c1] == None:
                    return None
                else:
                    raise ValueError("Two values")

            args[c1+c2] = df[[c1,c2]].apply(lambda x,c1=c1, c2=c2: switch(x,c1,c2), axis=1)

            args[c1+c2+"key"] = df[[c1,c2]].apply(lambda x,c1=c1, c2=c2: argswitch(x,c1,c2), axis=1)

            return df.assign(**args)

        super(MergeEmpty,self).__init__(fn, ['column1', 'column2'])



class Unfold(ParametrizedOperation):

    paramDescriptor = {'columnsKey': ParametrizedOperation.COLUMNS, 
                            'columnsTarget': ParametrizedOperation.COLUMNS}

    def __init__(self, columnsKey, columnsTarget):

        def fn(df, c1=columnsKey, c2=columnsTarget):

            keys = {}
            kcount = {}
            
            N = df.shape[0]
            for i in range(N): 

                k = tuple(df[c1].iloc[i,:])
                
                targetNull = True
                for a in df[c2].iloc[i,:]:
                    if a != None:
                        targetNull = False
                        break

                if k not in keys:
                    keys[k] = [None, None]
                    kcount[k] = 0

                kcount[k] += 1

                if kcount[k] > 2:
                    keys[k] = [None, None]
                    continue

                if targetNull and kcount[k] == 1:
                    keys[k][0] = i
                else:
                    keys[k][1] = i


            print(keys)
            dk = set(range(N))
            for k in keys:
                if keys[k][0] != None and keys[k][1] != None:
                    dk.remove(keys[k][1])
                    for c in c2:
                        ci = tuple(df.columns.values).index(c)
                        df.iloc[keys[k][0],ci] = df.iloc[keys[k][1],ci]

            print(dk)
            return df.iloc[list(dk)]

        super(Unfold,self).__init__(fn, ['columnsKey', 'columnsTarget'])



"""
Find an replace operation
"""
class Swap(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN, 
                                'predicate': ParametrizedOperation.PREDICATE,
                                'value': ParametrizedOperation.VALUE}

    def __init__(self, column, predicate, value):

        print("a,b", column, value)

        def fn(df, 
               column=column, 
               predicate=predicate, 
               v=value):

            N = df.shape[0]

            for i in range(N):
                if predicate(df.iloc[i,:]): #type issues
                    df[column].iloc[i] = v

            return df

        super(Swap,self).__init__(fn, ['column', 'predicate', 'value'])


"""
Find an replace operation
"""
class Delete(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN, 
                                'predicate': ParametrizedOperation.PREDICATE}

    def __init__(self, column, predicate):

        def fn(df, 
               column=column, 
               predicate=predicate):

            N = df.shape[0]

            for i in range(N):
                if predicate(df.iloc[i,:]): #type issues
                    df[column].iloc[i] = None

            return df

        super(Delete,self).__init__(fn, ['column', 'predicate'])

"""
Deletes a Column
"""
class DeleteColumn(ParametrizedOperation):

    paramDescriptor = {'column': ParametrizedOperation.COLUMN}

    def __init__(self, column):

        def fn(df, column=column):
            return df.drop(column, axis=1)

        super(DeleteColumn,self).__init__(fn, ['column'])


"""
No op
"""
class NOOP(Operation):

    def __init__(self):

        def fn(df):

            return df

        super(NOOP,self).__init__(fn)


"""
Force casts the data in the column to an integer
"""
class Cast(Operation):

    def __init__(self, attr, castFn, placeholder=None):

        def fn(df, 
               attr=attr, 
               castFn= castFn,
               placeholder=placeholder):

            N = df.shape[0]

            for i in range(N):

                try:
                    df[attr].iloc[i] = castFn(df[attr].iloc[i])
                except:
                    df[attr].iloc[i] = placeholder

            return df

        super(Cast,self).__init__(fn)


"""
Force casts the data in the column to an integer
"""
class Enumerate(Operation):

    def __init__(self):

        def fn(df):

            return df.assign(key=list(range(df.shape[0])))

        super(Enumerate,self).__init__(fn)




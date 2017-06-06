
"""
Core search routine
"""

from generators import *
from heapq import *
import numpy as np

def treeSearch(df, costFn, operations, evaluations=4):

    heap = []

    heappush(heap, ( np.sum(costFn(df)), NOOP() ) )

    for i in range(evaluations):

        value, op = heappop(heap)[0:2]
        p = ParameterSampler(df, costFn, operations)

        for opbranch in p.getAllOperations():

            nextop = op * opbranch

            nvalue = np.sum(costFn(nextop.run(df)))

            heappush(heap, (nvalue, nextop) )

            if nvalue == 0.0:
                return nextop

        heappush(heap, (value, op) )

    return heappop(heap)[1]

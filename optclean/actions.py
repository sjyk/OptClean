"""
This file defines all the allowed data cleaning actions
"""
import numpy as np
import copy

def getPossibleActions(dtype):
    if dtype == 'cat':
        return [{'fn': replaceAtRandom, 'params': 'value'}]

    if dtype == 'num':
        return [{'fn': around, 'params': None}, 
                {'fn': roundUp, 'params': None}, 
                {'fn': aroundDown, 'params': None}, 
                {'fn': replaceAtRandom, 'params': 'value'}]

def around(attr, row, params):
    row = row.copy(deep=True)
    row[attr] = np.around(row[attr])
    return row

def roundUp(attr, row, params):
    row = row.copy(deep=True)
    row[attr] = np.ceil(row[attr])
    return row

def roundDown(attr, row, params):
    row = row.copy(deep=True)
    row[attr] = np.floor(row[attr])
    return row

def replaceAtRandom(attr, row, params):
    row = row.copy(deep=True)
    row[attr] = params
    #print("--",row)
    #print(row, attr, params)
    return row
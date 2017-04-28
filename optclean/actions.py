"""
This file defines all the allowed data cleaning actions
"""
import numpy as np
import copy

def getPossibleActions(dtype):
    if dtype == 'cat':
        return [{'fn': replace, 'params': 'value', 'name': 'replace'}]

    if dtype == 'num':
        return [{'fn': around, 'params': None, 'name': 'around'}, 
                {'fn': roundUp, 'params': None, 'name': 'roundUp'}, 
                {'fn': roundDown, 'params': None, 'name': 'roundDown'}, 
                {'fn': replace, 'params': 'value', 'name': 'replace'}]

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

def replace(attr, row, params):
    row = row.copy(deep=True)
    row[attr] = params
    #print("--",row)
    #print(row, attr, params)
    return row
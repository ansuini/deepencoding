'''
Collection of useful small functions
'''

from operator import mul

def productory(list):
    return reduce(mul, list)

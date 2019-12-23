"""
Created on Tue Nov 26 14:56:47 2019

@author: Jing Yang (jyang7@fsu.edu) and Ming Ye (mye@fsu.edu). 
The MIT License (MIT)

Copyright (c) 2019 Jing Yang and Ming Ye

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

# Non-monotonic Sobol G Function (9 parameters)
def evaluate(values, a=None):
    '''
    The Sobol G function is described in the reference below:
    Francesca Campolongo, Jessica Cariboni, Andrea Saltelli,
    An effective screening design for sensitivity analysis of large models,
    Environmental Modelling & Software,
    Volume 22, Issue 10,
    2007,
    Pages 1509-1518,
    ISSN 1364-8152,
    https://doi.org/10.1016/j.envsoft.2006.10.004.
    
    Parameters:
    values: a matrix to store parameter values of the N random trajectories
    a: this vector is specifically for the Sobol G function.
    
    Returns
    -------
    Y: The model run outputs. A column vector with the dimension of (k+1)xN
    ----------
    '''
    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
    if a is None:
        a = [0.02, 0.03, 0.05, 11, 12.5, 13, 34, 35, 37]

    ltz = values < 0
    gto = values > 1

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than zero")
    elif gto.any() == True:
        raise ValueError("Sobol G function called with values greater than one")

    Y = np.ones([values.shape[0]])

    len_a = len(a)
    for i, row in enumerate(values):
        for j in range(len_a):
            x = row[j]
            a_j = a[j]
            Y[i] *= (np.abs(4 * x - 2) + a_j) / (1 + a_j)

    return Y

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:06:20 2019

@author: Jing Yang (jyang7@fsu.edu) and Ming Ye (mye@fsu.edu). This program was modified from the python codes
available at https://salib.readthedocs.io/en/latest/api.html, but it supports generation of random trajectories for parameters 
following the uniform (unif), normal (norm), lognormal (lognorm), triangular (triang), and norm distribution truncated above zero (truncnorm) distributions.

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
import scipy.stats
from collections import OrderedDict

def scale_samples(params, bounds):
    '''Rescale samples in 0-to-1 range to arbitrary bounds

    Arguments
    ---------
    bounds : list
        list of lists of dimensions `num_params`-by-2
    params : numpy.ndarray
        numpy array of dimensions `num_params`-by-:math:`N`,
        where :math:`N` is the number of samples
    '''
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(params,
                       (upper_bounds - lower_bounds),
                       out=params),
           lower_bounds,
           out=params)


def unscale_samples(params, bounds):
    """Rescale samples from arbitrary bounds back to [0,1] range

    Arguments
    ---------
    bounds : list
        list of lists of dimensions num_params-by-2
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    """
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   (sample - lower_bound) / (upper_bound - lower_bound)
    np.divide(np.subtract(params, lower_bounds, out=params),
              np.subtract(upper_bounds, lower_bounds),
              out=params)


def nonuniform_scale_samples(params, bounds, dists):
    """Rescale samples in 0-to-1 range to other distributions

    Arguments
    ---------
    problem : dict
        problem definition including bounds
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    dists : list
        list of distributions, one for each parameter
            unif: uniform with lower and upper bounds
            triang: triangular with width (scale) and location of peak
                    location of peak is in percentage of width
                    lower bound assumed to be zero
            norm: normal distribution with mean and standard deviation
            lognorm: lognormal with ln-space mean and standard deviation
    """
    b = np.array(bounds)

    # initializing matrix for converted values
    conv_params = np.zeros_like(params)

    # loop over the parameters
    for i in range(conv_params.shape[1]):
        # setting first and second arguments for distributions
        b1 = b[i][0]
        b2 = b[i][1]

        if dists[i] == 'triang':
            # checking for correct parameters
            if b1 <= 0 or b2 <= 0 or b2 >= 1:
                raise ValueError('''Triangular distribution: Scale must be
                    greater than zero; peak on interval [0,1]''')
            else:
                conv_params[:, i] = scipy.stats.triang.ppf(
                    params[:, i], c=b2, scale=b1, loc=0)

        elif dists[i] == 'unif':
            if b1 >= b2:
                raise ValueError('''Uniform distribution: lower bound
                    must be less than upper bound''')
            else:
                conv_params[:, i] = params[:, i] * (b2 - b1) + b1

        elif dists[i] == 'norm':
            if b2 <= 0:
                raise ValueError('''Normal distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = scipy.stats.truncnorm.ppf(
                    params[:, i], scipy.stats.norm.ppf(0.0001,  \
                          loc=b1, scale=b2), scipy.stats.norm.ppf(0.9999,  \
                          loc=b1, scale=b2), loc=b1, scale=b2)

        # lognormal distribution (ln-space, not base-10)
        # paramters are ln-space mean and standard deviation
        elif dists[i] == 'lognorm':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Lognormal distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = np.exp(scipy.stats.truncnorm.ppf(
                    params[:, i], scipy.stats.norm.ppf(0.0001,  \
                          loc=b1, scale=b2), scipy.stats.norm.ppf(0.9999,  \
                          loc=b1, scale=b2), loc=b1, scale=b2))
                
        # truncnormal distribution (ln-space, not base-10)
        # paramters are ln-space mean and standard deviation truncated at 0
        elif dists[i] == 'truncnorm':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Truncnorm distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = scipy.stats.truncnorm.ppf(
                    params[:, i], (0 - b1) / b2, scipy.stats.norm.ppf(0.9999,  \
                          loc=b1, scale=b2), loc=b1, scale=b2)
                
        else:
            valid_dists = ['unif', 'triang', 'norm', 'lognorm', 'truncnorm']
            raise ValueError('Distributions: choose one of %s' %
                             ", ".join(valid_dists))

    return conv_params

def compute_groups_matrix(groups):
    """Generate matrix which notes factor membership of groups

    Computes a k-by-g matrix which notes factor membership of groups
    where:
        k is the number of variables (factors)
        g is the number of groups
    Also returns a g-length list of unique group_names whose positions
    correspond to the order of groups in the k-by-g matrix

    Arguments
    ---------
    groups : list
        Group names corresponding to each variable

    Returns
    -------
    tuple
        containing group matrix assigning parameters to
        groups and a list of unique group names
    """
    if not groups:
        return None

    num_vars = len(groups)

    # Get a unique set of the group names
    unique_group_names = list(OrderedDict.fromkeys(groups))
    number_of_groups = len(unique_group_names)

    indices = dict([(x, i) for (i, x) in enumerate(unique_group_names)])

    output = np.zeros((num_vars, number_of_groups), dtype=np.int)

    for parameter_row, group_membership in enumerate(groups):
        group_index = indices[group_membership]
        output[parameter_row, group_index] = 1

    return output, unique_group_names

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:56:47 2019

@author: Jing Yang (jyang7@fsu.edu) and Ming Ye (mye@fsu.edu). This program was modified from the python codes
available at https://salib.readthedocs.io/en/latest/api.html, but has the following features:
(1) it supports generation of random trajectories for parameters following the following distributions: 
    uniform (unif), normal (norm), lognormal (lognorm), # triangular (triang), and norm distribution truncated above zero (truncnorm).
(2) it can be used for the level number other than 4. The original python code seems to have a bug that it only works with the level number of 4.

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
from __future__ import division

import numpy as np
from util import scale_samples, nonuniform_scale_samples, compute_groups_matrix

def sample(parameter_space, N, num_levels=4, seed=None):
    """Generate N trajectories using the Method of Morris
    The details of the random trajectory generation are referred to Section 3.3 of 
    the book entitled "Global Sensitivity Analysis: The Primer" by saltelli et al. (2007)

    Parameters
    ----------
    parameter_space : dictionary
        The parameter_space definition
    N : integer
        The number of trajectories to generate
    num_levels : integer, default=4
        The number of grid levels (should be even)

    Returns
    -------
    sample or scaled_samples: numpy.ndarray
        Returns a numpy.ndarray matrix containing the N trajectories required for 
        Morris sensitivity analysis. Each trajectory (denoted as B* in Section 3.3 of Saltelli's book) 
        has the dimension of (k+1) x k, where k is the number of model parameters considered in the 
        sensitivit analysis. All the trajectories are included one after anotehr in the returned matrix. 
        As a result, the dimension of the returned matrix is ((k+1)xN) x k. When applying the Morris method
        to grouped parameters, the matrix dimension changes, and the details are referred to Section 3.5 of
        Saltellie's book.
    """
    assert (num_levels%2 == 0), 'num_levels must be even'
    
    if seed:                    # generate random seed if users do not provide a seed
        np.random.seed(seed)

# if parameters are grouped, use function sample_groups; otherwise, use function sample_oat (one parameter at a time)        
    if parameter_space.get('groups'): 
        samples = _sample_groups(parameter_space, N, num_levels)
    else:
        samples = _sample_oat(parameter_space, N, num_levels)

# The function scale_samples is used only when the 'dists" keyword is not used when defining the parameter space.
    if not parameter_space.get('dists'):
        # scaling values out of 0-1 range with uniform distributions
        scale_samples(samples, parameter_space['bounds'])
        return samples
    else:
        # scaling values to other distributions based on inverse CDFs
        scaled_samples = nonuniform_scale_samples(samples, parameter_space['bounds'], parameter_space['dists'])
        return scaled_samples


def _sample_oat(parameter_space, N, num_levels=4):
    """Generate trajectories without groups

    Arguments
    ---------
    parameter_space : dict
        The parameter_space definition
    N : int
        The number of samples to generate
    num_levels : int, default=4
        The number of grid levels
    """
    group_membership = np.asmatrix(np.identity(parameter_space['num_vars'],
                                               dtype=int))

    num_params = group_membership.shape[0]
    
    sample = np.array([generate_trajectory(group_membership,
                                           num_levels)
                       for n in range(N)])
    return sample.reshape((N * (num_params + 1), num_params))


def _sample_groups(parameter_space, N, num_levels=4):
    """Generate trajectories for groups

    Returns an :math:`N(g+1)`-by-:math:`k` array of `N` trajectories,
    where :math:`g` is the number of groups and :math:`k` is the number
    of factors

    Arguments
    ---------
    parameter_space : dict
        The parameter_space definition
    N : int
        The number of trajectories to generate
    num_levels : int, default=4
        The number of grid levels

    Returns
    -------
    numpy.ndarray
    """
    if len(parameter_space['groups']) != parameter_space['num_vars']:
        raise ValueError("Groups do not match to number of variables")

    group_membership, _ = compute_groups_matrix(parameter_space['groups'])

    if group_membership is None:
        raise ValueError("Please define the 'group_membership' matrix")
    if not isinstance(group_membership, np.ndarray):
        raise TypeError("Argument 'group_membership' should be formatted \
                         as a numpy ndarray")

    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]
    sample = np.zeros((N * (num_groups + 1), num_params))
    sample = np.array([generate_trajectory(group_membership,
                                           num_levels)
                       for n in range(N)])
    return sample.reshape((N * (num_groups + 1), num_params))


def generate_trajectory(group_membership, num_levels=4):
    """Return a single trajectory

    Return a single trajectory of size :math:`(g+1)`-by-:math:`k`
    where :math:`g` is the number of groups,
    and :math:`k` is the number of factors,
    both implied by the dimensions of `group_membership`

    Arguments
    ---------
    group_membership : np.ndarray
        a k-by-g matrix which notes factor membership of groups
    num_levels : int, default=4
        The number of levels in the grid

    Returns
    -------
    np.ndarray
    """

    delta = compute_delta(num_levels)

    # Infer number of groups `g` and number of params `k` from
    # `group_membership` matrix
    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]

    # Matrix B - size (g + 1) * g -  lower triangular matrix
    B = np.tril(np.ones([num_groups + 1, num_groups],
                        dtype=int), -1)

    P_star = generate_p_star(num_groups)

    # Matrix J - a (g+1)-by-num_params matrix of ones
    J = np.ones((num_groups + 1, num_params))

    # Matrix D* - num_params-by-num_params matrix which decribes whether
    # factors move up or down
    D_star = np.diag(np.random.choice([-1, 1], num_params))

    x_star = generate_x_star(num_params, num_levels)

    # Matrix B* - size (num_groups + 1) * num_params
    B_star = compute_b_star(J, x_star, delta, B,
                            group_membership, P_star, D_star)
    
    return B_star


def compute_b_star(J, x_star, delta, B, G, P_star, D_star):
    """
    """
    element_a = J[0, :] * x_star
    element_b = np.matmul(G, P_star).T
    element_c = np.matmul(2.0 * B, element_b)
    element_d = np.matmul((element_c - J), D_star)

    b_star = element_a + (delta / 2.0) * (element_d + J)
    return b_star


def generate_p_star(num_groups):
    """Describe the order in which groups move

    Arguments
    ---------
    num_groups : int

    Returns
    -------
    np.ndarray
        Matrix P* - size (g-by-g)
    """
    p_star = np.eye(num_groups, num_groups)
    np.random.shuffle(p_star)
    return p_star


def generate_x_star(num_params, num_levels):
    """Generate an 1-by-num_params array to represent initial position for EE

    This should be a randomly generated array in the p level grid
    :math:`\omega`

    Arguments
    ---------
    num_params : int
        The number of parameters (factors)
    num_levels : int
        The number of levels

    Returns
    -------
    numpy.ndarray
        The initial starting positions of the trajectory

    """
    x_star = np.zeros((1, num_params))
    delta = compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0, bound, int(num_levels / 2))

    x_star[0, :] = np.random.choice(grid, num_params)

    return x_star

def compute_delta(num_levels):
    """Computes the delta value from number of levels

    Arguments
    ---------
    num_levels : int
        The number of levels

    Returns
    -------
    float
    """
    return num_levels / (2.0 * (num_levels - 1))

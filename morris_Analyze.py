# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:58:43 2019

@author: Jing Yang (jyang7@fsu.edu) and Ming Ye (mye@fsu.edu). This program was modified from the python codes
available at https://salib.readthedocs.io/en/latest/api.html, but it can be used for the level number other than 4 (the original python code 
seems to have a bug that it only works with the level number of 4). In addition, all the elementary effects are returned to the main program
for convergence plotting.

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
from morris_Sample import compute_delta
from results import ResultDict
from util import compute_groups_matrix

def analyze(parameter_space, X, Y,
            num_levels,
            print_to_console=True):
    """Perform Morris Analysis on model outputs.

    Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and
    'mu_star_conf', where each entry is a list of parameters containing
    the indices in the same order as the parameter file.

    Arguments
    ---------
    parameter_space : dict
        The parameter_space definition
    X : numpy.matrix
        The NumPy matrix containing the model inputs of dtype=float
    Y : numpy.array
        The NumPy array containing the model outputs of dtype=float
    num_levels : int
        The number of grid levels, must be identical to the value
        passed to SALib.sample.morris (default 4)

    Returns
    -------
    ee : the matrix of elementrary effects
    Si : dict
        A dictionary of sensitivity indices containing the following entries.

        - `mu` - the mean elementary effect
        - `mu_star` - the absolute of the mean elementary effect
        - `sigma` - the standard deviation of the elementary effect
        - `names` - the names of the parameters

    References
    ----------
    .. [1] Morris, M. (1991).  "Factorial Sampling Plans for Preliminary
           Computational Experiments."  Technometrics, 33(2):161-174,
           doi:10.1080/00401706.1991.10484804.
    .. [2] Campolongo, F., J. Cariboni, and A. Saltelli (2007).  "An effective
           screening design for sensitivity analysis of large models."
           Environmental Modelling & Software, 22(10):1509-1518,
           doi:10.1016/j.envsoft.2006.10.004.

    """
 
    msg = ("dtype of {} array must be 'float', float32 or float64")
    if X.dtype not in ['float', 'float32', 'float64']:
        raise ValueError(msg.format('X'))
    if Y.dtype not in ['float', 'float32', 'float64']:
        raise ValueError(msg.format('Y'))

    # Assume that there are no groups
    groups = None
    delta = compute_delta(num_levels)
    
    num_vars = parameter_space['num_vars']

    if (parameter_space.get('groups') is None) & (Y.size % (num_vars + 1) == 0):
        num_trajectories = int(Y.size / (num_vars + 1))
    elif parameter_space.get('groups') is not None:
        groups, unique_group_names = compute_groups_matrix(
            parameter_space['groups'])
        number_of_groups = len(unique_group_names)
        num_trajectories = int(Y.size / (number_of_groups + 1))
    else:
        raise ValueError("Number of samples in model output file must be"
                         "a multiple of (D+1), where D is the number of"
                         "parameters (or groups) in your parameter file.")
    ee = np.zeros((num_vars, num_trajectories))
    ee = compute_elementary_effects(
        X, Y, int(Y.size / num_trajectories), delta)

    # Output the Mu, Mu*, and Sigma Values. Also return them in case this is
    # being called from Python
    Si = ResultDict((k, [None] * num_vars)
                    for k in ['names', 'mu', 'mu_star', 'sigma'])
    Si['mu'] = np.average(ee, 1)
    Si['mu_star'] = np.average(np.abs(ee), 1)
    Si['sigma'] = np.std(ee, axis=1, ddof=1)
    Si['names'] = parameter_space['names']

    if groups is None:
        if print_to_console:
            print("{0:<30} {1:>10} {2:>10} {3:>15}".format(
                "Parameter",
                "Mu_Star",
                "Mu",
                "Sigma")
            )
            for j in list(range(num_vars)):
                print("{0:30} {1:10.3f} {2:10.3f} {3:15.3f}".format(
                    Si['names'][j],
                    Si['mu_star'][j],
                    Si['mu'][j],
                    Si['sigma'][j])
                )
        return ee, Si
    elif groups is not None:
        # if there are groups, then the elementary effects returned need to be
        # computed over the groups of variables,
        # rather than the individual variables
        Si_grouped = dict((k, [None] * num_vars)
                          for k in ['mu_star', 'mu_star_conf'])
        Si_grouped['mu_star'] = compute_grouped_metric(Si['mu_star'], groups)
        Si_grouped['names'] = unique_group_names
        Si_grouped['sigma'] = compute_grouped_sigma(Si['sigma'], groups)
        Si_grouped['mu'] = compute_grouped_sigma(Si['mu'], groups)

        if print_to_console:
            print("{0:<30} {1:>10} {2:>10} {3:>15}".format(
                "Parameter",
                "Mu_Star",
                "Mu",
                "Sigma")
            )
            for j in list(range(number_of_groups)):
                print("{0:30} {1:10.3f} {2:10.3f} {3:15.3f}".format(
                    Si_grouped['names'][j],
                    Si_grouped['mu_star'][j],
                    Si_grouped['mu'][j],
                    Si_grouped['sigma'][j])
                )

        return ee, Si_grouped
    else:
        raise RuntimeError(
            "Could not determine which parameters should be returned")


def compute_grouped_sigma(ungrouped_sigma, group_matrix):
    '''
    Returns sigma for the groups of parameter values in the
    argument ungrouped_metric where the group consists of no more than
    one parameter
    '''

    group_matrix = np.array(group_matrix, dtype=np.bool)

    sigma_masked = np.ma.masked_array(ungrouped_sigma * group_matrix.T,
                                      mask=(group_matrix ^ 1).T)
    sigma_agg = np.ma.mean(sigma_masked, axis=1)
    sigma = np.zeros(group_matrix.shape[1], dtype=np.float)
    np.copyto(sigma, sigma_agg, where=group_matrix.sum(axis=0) == 1)
    np.copyto(sigma, np.NAN, where=group_matrix.sum(axis=0) != 1)

    return sigma



def compute_grouped_metric(ungrouped_metric, group_matrix):
    '''
    Computes the mean value for the groups of parameter values in the
    argument ungrouped_metric
    '''

    group_matrix = np.array(group_matrix, dtype=np.bool)

    mu_star_masked = np.ma.masked_array(ungrouped_metric * group_matrix.T,
                                        mask=(group_matrix ^ 1).T)
    mean_of_mu_star = np.ma.mean(mu_star_masked, axis=1)

    return mean_of_mu_star



def get_increased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (1, 0), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (0, 1), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T



def get_decreased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (0, 1), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (1, 0), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T



def compute_elementary_effects(model_inputs, model_outputs, trajectory_size,
                               delta):
    '''
    Arguments
    ---------
    model_inputs : matrix of inputs to the model under analysis.
        x-by-r where x is the number of variables and
        r is the number of rows (a function of x and num_trajectories)
    model_outputs
        an r-length vector of model outputs
    trajectory_size
        a scalar indicating the number of rows in a trajectory
    delta : float
        scaling factor computed from `num_levels`

    Returns
    ---------
    ee : np.array
        Elementary Effects for each parameter
    '''
    num_vars = model_inputs.shape[1]
    num_rows = model_inputs.shape[0]
    num_trajectories = int(num_rows / trajectory_size)

    ee = np.zeros((num_trajectories, num_vars), dtype=np.float)

    ip_vec = model_inputs.reshape(num_trajectories, trajectory_size, num_vars)
    ip_cha = np.subtract(ip_vec[:, 1:, :], ip_vec[:, 0:-1, :])
    up = (ip_cha > 0)
    lo = (ip_cha < 0)

    op_vec = model_outputs.reshape(num_trajectories, trajectory_size)

    result_up = get_increased_values(op_vec, up, lo)
    result_lo = get_decreased_values(op_vec, up, lo)

    ee = np.subtract(result_up, result_lo)
    np.divide(ee, delta, out=ee)

    return ee

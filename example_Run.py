# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:15:47 2019

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
import matplotlib.pyplot as plt

#
# Step 1: Generate random trajectories in the parameter space of the Sobol G function
#

from morris_Sample import sample

# Define the parameter space of the Sobol G function. The function has nine parameters, 
# called x1, x2, ..., x9. Each parameter follows a uniform distribution U[0,1]. In addition, 
# the nince parameters belong to nine groups. The user can change the parameter number, names, 
# bounds, distributions, and group classification. Our program supports random trajectory generation 
# for the following distributions: uniform (unif), normal (norm), lognormal (lognorm), 
# triangular (triang), and norm distribution truncated above zero (truncnorm).  

# When defining the parameter sapce, the "groups" and "dists" keywords are optional. 
# If "groups" is not used, the program assumes that each parameter belongs to a group.
# If "dists" is not used, the program assumes that each parameter follows a uniform distribution. 
# However, the parameter bounds must be specified in "bounds". 
parameter_space = {'num_vars': 9,
           'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',  'x8', 'x9'],
           # 'groups': ['Group0', 'Group0', 'Group0', 'Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group8'],
           'bounds': [[0.0, 1.0]] * 9,
           'dists': ['unif'] * 9
           }

# Generate samples. In this example, NNP trajectories are generated, and the number of level is set as 6. In reality,
# the number of trajectories should be significantly smaller to save computational cost. The users can adjust 
# the number of levels and seed value. The output, param_values, is a matrix with the dimension of ((k+1)xN) x k, 
# where k (9 in this example) is the number of model parameters considered in the sensitivity analysis, and N (NNP in this example)
# is the number of random trajectories. 
NNP=1000
param_values = sample(parameter_space, NNP, 6, seed=110287)

#
# Step 2: Evaluate Sobol G function for the random parameter saples generated in Step 1 above
# 

#The file sobol_G.py defines and evaluates the Sobol G function. 
from sobol_G import evaluate

# Users need to run their own models by using the random paramters as inputs. The model run outputs, called Y in this file,
# should be organized in a column vector with the dimension of (k+1)xN
Y = evaluate(param_values)

#
# Step 3: Calculate elementrary effects and their mean and standard deviation
# 

# The analyze function computes elementrary effects (ee) and their statistis (Si)
from morris_Analyze import analyze
ee, Si = analyze(parameter_space , param_values, Y, 6)

# Plot the mean vs. standard variation
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
for i in range(3):
    plt.scatter(Si['mu_star'][i], Si['sigma'][i])
    plt.text(Si['mu_star'][i], Si['sigma'][i], parameter_space['names'][i])
    plt.xlim([0, 3])
    plt.ylim([0, 5]) 
    plt.xlabel('mu_star')
    plt.ylabel('sigma')

plt.subplot(1, 3, 2)
for i in range(3, 6):
    plt.scatter(Si['mu_star'][i], Si['sigma'][i])
    plt.text(Si['mu_star'][i], Si['sigma'][i], parameter_space['names'][i])
    plt.xlim([0, 3])
    plt.ylim([0, 5]) 
    plt.xlabel('mu_star')
    plt.ylabel('sigma')

plt.subplot(1, 3, 3)
for i in range(6, 8):
    plt.scatter(Si['mu_star'][i], Si['sigma'][i])
    plt.text(Si['mu_star'][i], Si['sigma'][i], parameter_space['names'][i])
    plt.xlim([0, 3])
    plt.ylim([0, 5])
    plt.xlabel('mu_star')
    plt.ylabel('sigma')
    
plt.show()

#
# Plot convergences
#
# This part of code is to evaluate convergence of the mean and standard deviation 
# when the number (r) of random trajectories increases. This is to help determine an
# appropriate r value.

mean_abs_ee = np.zeros([9, NNP])
var_ee = np.zeros([9, NNP])

for i in range(9):
    for j in range (NNP):
        mean_abs_ee[i, j] = np.mean(abs(ee[i, :j]))
        var_ee[i, j] = np.var(ee[i, :j])
        
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for i in range(9):
    plt.plot(mean_abs_ee[i, :])
plt.xlim([0, NNP])    
plt.xlabel('Number of trajectories')
plt.ylabel('mu_star')

plt.subplot(1, 2, 2)     
for i in range(9):
    plt.plot(var_ee[i, :])
plt.xlim([0, NNP])    
plt.xlabel('Number of trajectories')
plt.ylabel('sigma')

plt.show()




    
        
    
    




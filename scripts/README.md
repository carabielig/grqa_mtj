# scripts

### grqa

This Python package contains all the code for the Recurrence Quantification Analysis, 
including the line sampling and patch sampling approach, as well as the mass distribution.

#### grqa_cpu

This file contains very slow and basic functions for the line sampling and patch sampling
approach. They can return the indices of lines and patches and are therefore great to 
understand the approach as well as for testing.

#### grqa_cpu_fast

This file contains improved versions of the line and patch sampling approach as well as the mass
distribution in two versions, one more and one less memory efficient.

#### grqa_gpu

This file includes the line sampling approach as well as mass distribution (including an approach
for drawing random boxes from the recurrence plot and the approach for storing intermediate results
and calculating larger box sizes from these intermediate results) for the GPU.

#### grqa_measures

This file contains the calculation of the Recurrence Quantification Analysis measures from the results of the line
and patch sampling approach as well as from the mass distribution.


### utils

#### plot

Contains functions for plotting, mainly for plotting a 3d subset of the 4d recurrence plots.

#### test_images

This file contains multiple functions for the creation of test patterns (e.g. uniformly distributed 
white noise, gaussian white noise, turing pattern, fractal, logistic map, periodic pattern,
longitudinal wave).

#### turing

Code for the creation of a turing pattern. Code was taken from: 
https://github.com/ijmbarr/turing-patterns/blob/master/turing-patterns.ipynb (02-2025)  
https://github.com/ijmbarr/turing-patterns/blob/master/tutils.py (02-2025)

#### utils

Contains functions to preprocess and load data, as well as other helpful helper functions.


### other files

#### thesis_figures

Plot figures for the thesis. Contains the code for generating the example figures in the theoretical
background chapter of the thesis.
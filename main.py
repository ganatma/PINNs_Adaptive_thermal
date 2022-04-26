#%% Improt required packages


import scipy.io
import math
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import *

#%% Define Domain

Domain = DomainND(["x", "y", "t"], time_var="t")

Domain.add("x", [0, 1.0], 101)
Domain.add("y", [0, 1.0], 101)
Domain.add("t", [0, 6.0], 110)

N_f = 100000
Domain.generate_collocation_points(N_f)

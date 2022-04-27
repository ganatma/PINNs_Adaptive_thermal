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

# %% Defining function for IC's and BC's


def func_ic(x, y):
    return 300.0


def fun_x(x):
    return 0.0


def fun_y(y):
    return 0.0


init = IC(Domain, [func_ic], var=[["x", "y"]])


def deriv_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    return u_x, u_y


upper_x = FunctionNeumannBC(
    Domain,
    fun=[fun_x],
    var="x",
    target="upper",
    deriv_model=[deriv_model],
    func_inputs=["x"],
    n_values=5,
)

upper_y = FunctionNeumannBC(
    Domain,
    fun=[fun_y],
    var="y",
    target="upper",
    deriv_model=[deriv_model],
    func_inputs=["y"],
    n_values=25,
)

lower_x = FunctionNeumannBC(
    Domain,
    fun=[fun_x],
    var="x",
    target="lower",
    deriv_model=[deriv_model],
    func_inputs=["x"],
    n_values=25,
)

lower_y = FunctionNeumannBC(
    Domain,
    fun=[fun_y],
    var="y",
    target="lower",
    deriv_model=[deriv_model],
    func_inputs=["y"],
    n_values=25,
)

BCs = [init, upper_x, lower_x, upper_y, lower_y]

#%% Define f model for residual equations


def f_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    u_y = tf.gradients(u, y)
    u_yy = tf.gradients(u_y, y)
    coeff_1 = tdq.utils.constant(6.0)
    coeff_2 = tdq.utils.constant(math.sqrt(3 / (math.pow(math.pi, 3))))
    coeff_3 = tdq.utils.constant((0.7 * 900.0) / (1e-8))
    rho = tdq.utils.constant(8000.0)
    cp = tdq.utils.constant(500.0)
    k = tdq.utils.constant(16.3)
    heat_source_coeff = coeff_1 * coeff_2 * coeff_3
    fx = 1.8e-1 * t
    c1 = (x - fx) / (1e-4)
    c2 = (y - 0.5) / (1e-4)
    res = (
        (rho * cp * u_t)
        - (k * (u_xx + u_yy))
        - (heat_source_coeff * tf.math.exp((-3.0) * (c1 ** 2) + (c2 ** 2)))
    )
    return res


# Define which loss functions will have adaptive weigths
dict_adaptive = {"residual": [True], "BCs": [True, False, False, False, False]}

# Weigths Initialization

init_weights = {
    "residual": [tf.random.uniform([N_f, 1])],
    "BCs": [100 * tf.random.uniform([101, 1]), None, None, None, None],
}

# Define NN architecture and intialize model
layer_sizes = [3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(
    layer_sizes,
    f_model,
    Domain,
    BCs,
    isAdaptive=True,
    dict_adaptive=dict_adaptive,
    init_weights=init_weights,
)

model.fit(tf_iter=10000, newton_iter=10000)
# %%

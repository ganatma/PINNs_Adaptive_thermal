#%% Import required libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs

#%% Defining size of network

layer_sizes = [3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]

for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))


# L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0
def set_weights(model, w, sizes_w, sizes_b):
    for i, layer in enumerate(model.layers[0:]):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[: i + 1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights : end_weights + sizes_b[i]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)


def get_weights(model):
    w = []
    for layer in model.layers[0:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)

    w = tf.convert_to_tensor(w)
    return w


# define the neural network model
def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(
            layers.Dense(
                width, activation=tf.nn.tanh, kernel_initializer="glorot_normal"
            )
        )
    model.add(
        layers.Dense(
            layer_sizes[-1], activation=None, kernel_initializer="glorot_normal"
        )
    )
    return model


#%% Define loss


def loss(
    x_data_batch,
    y_data_batch,
    t_data_batch,
    T_data_batch,
    x_0,
    y_0,
    t_0,
    T_0,
    x_colloc_batch,
    y_colloc_batch,
    t_colloc_batch,
    x_ulb,
    t_ulb,
    T_ulb,
    y_lrb,
    t_lrb,
    T_lrb,
    weights_data,
    weights_0,
    weights_colloc,
    weights_b,
):
    # Loss_data component
    T_data_pred = u_model(x_data_batch, y_data_batch, t_data_batch)
    mse_data = tf.reduce_mean(tf.square(weights_data * (T_data_batch - T_data_pred)))

    # Loss_0 component
    T_0_pred = u_model(x_0, y_0, t_0)
    mse_0 = tf.reduce_mean(tf.square(weights_0 * (T_0 - T_0_pred)))

    # Loss_colloc component
    f_T_pred = f_model(x_colloc_batch, y_colloc_batch, t_colloc_batch)
    mse_colloc = tf.reduce_mean(tf.square(weights_colloc * f_T_pred))

    # Loss_b component
    T_y_ulb_pred = u_derv_model(u_model, x_ulb, None)
    T_x_lrb_pred = u_derv_model(u_model, None, y_lrb)
    mse_b = tf.reduce_mean(tf.square(weights_b * T_y_ulb_pred)) + tf.reduce_mean(
        tf.square(weights_b * T_x_lrb_pred)
    )

    mse = mse_data + mse_0 + mse_colloc + mse_b

    return mse

#%% Define f_model and u_model

@tf.function
def f_model(x,y,t):
    T = u_model(x,y,t)
    T_x = tf.gradients(T,x)[0]
    T_y = tf.gradients(T,y)[0]
    T_xx = tf.gradients(T_x,x)[0]
    T_yy = tf.gradients(T_y,y)[0]

    coeff_1 = tf.constant(6.0, dtype=tf.float32)
    coeff_2 = tf.constant(math.sqrt(3 / (math.pow(math.pi, 3))), dtype=tf.float32)
    coeff_3 = (nu * P) / (r ** 2)
    heat_source_coeff = coeff_1 * coeff_2 * coeff_3
    fx = 1.8 * (10 ** (-1)) * t
    c1 = (x - fx) / r
    c2 = (y - 0.5) / r

    residual = (
        rho * cp * u_t
        - k * (u_xx + u_yy)
        - heat_source_coeff * tf.math.exp((-3.0) * (c1 ** 2 + c2 ** 2))
    )

    return residual


#%%

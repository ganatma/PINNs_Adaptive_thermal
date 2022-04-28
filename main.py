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

sizes_w = []
sizes_b = []
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


u_model = neural_net(layer_sizes)
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
    y_ulb,
    t_ulb,
    x_lrb,
    y_lrb,
    t_lrb,
    weights_data,
    weights_0,
    weights_colloc,
    weights_ulb,
    weights_lrb,
):
    # Loss_data component
    T_data_pred = u_model(tf.concat([x_data_batch, y_data_batch, t_data_batch], 1))
    mse_data_sa = tf.reduce_mean(tf.square(weights_data * (T_data_batch - T_data_pred)))
    mse_data = tf.reduce_mean(tf.square((T_data_batch - T_data_pred)))

    # Loss_0 component
    T_0_pred = u_model(tf.concat([x_0, y_0, t_0], 1))
    mse_0_sa = tf.reduce_mean(tf.square(weights_0 * (T_0 - T_0_pred)))
    mse_0 = tf.reduce_mean(tf.square((T_0 - T_0_pred)))

    # Loss_colloc component
    f_T_pred = f_model(x_colloc_batch, y_colloc_batch, t_colloc_batch)
    mse_colloc_sa = tf.reduce_mean(tf.square(weights_colloc * f_T_pred))
    mse_colloc = tf.reduce_mean(tf.square(f_T_pred))

    # Loss_b component
    T_y_ulb_pred = u_derv_ulb_model(u_model, x_ulb, y_ulb, t_ulb)
    T_x_lrb_pred = u_derv_lrb_model(u_model, x_lrb, y_lrb, t_lrb)
    mse_ulb_sa = tf.reduce_mean(tf.square(weights_ulb * T_y_ulb_pred))
    mse_ulb = tf.reduce_mean(tf.square(T_y_ulb_pred))
    mse_lrb_sa = tf.reduce_mean(tf.square(weights_lrb * T_x_lrb_pred))
    mse_lrb = tf.reduce_mean(tf.square(T_x_lrb_pred))

    mse = mse_data_sa + mse_0_sa + mse_colloc_sa + mse_ulb_sa + mse_lrb_sa

    return mse, mse_data, mse_0, mse_colloc, mse_ulb, mse_lrb


#%% Define f_model and u_derv_model


@tf.function
def f_model(x, y, t):
    T = u_model(tf.concat([x, y, t], 1))
    T_x = tf.gradients(T, x)[0]
    T_y = tf.gradients(T, y)[0]
    T_xx = tf.gradients(T_x, x)[0]
    T_yy = tf.gradients(T_y, y)[0]
    T_t = tf.gradients(T, t)[0]

    rho = tf.constant(8000.0, dtype=tf.float32)
    cp = tf.constant(500.0, dtype=tf.float32)
    k = tf.constant(16.3, dtype=tf.float32)
    P = tf.constant(900.0, dtype=tf.float32)
    nu = tf.constant(0.7, dtype=tf.float32)
    r = tf.constant(1e-4, dtype=tf.float32)
    coeff_1 = tf.constant(6.0, dtype=tf.float32)
    coeff_2 = tf.constant(math.sqrt(3 / (math.pow(math.pi, 3))), dtype=tf.float32)
    coeff_3 = tf.constant((nu * P) / (r ** 2), dtype=tf.float32)
    heat_source_coeff = coeff_1 * coeff_2 * coeff_3
    fx = (1.8e-1) * t
    c1 = (x - fx) / r
    c2 = (y - 0.5) / r

    residual = (
        rho * cp * T_t
        - k * (T_xx + T_yy)
        - heat_source_coeff * tf.math.exp((-3.0) * (c1 ** 2 + c2 ** 2))
    )

    return residual


def u_derv_ulb_model(u_model, x, y, t):
    T = u_model(tf.concat([x, y, t], 1))
    return tf.gradients(T, y)[0]


def u_derv_lrb_model(u_model, x, y, t):
    T = u_model(tf.concat([x, y, t], 1))
    return tf.gradients(T, x)[0]


#%% grad function for training of variables


def grad(
    model,
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
    y_ulb,
    t_ulb,
    x_lrb,
    y_lrb,
    t_lrb,
    weights_data,
    weights_0,
    weights_colloc,
    weights_ulb,
    weights_lrb,
):

    with tf.GradientTape(persistent=True) as tape:
        loss_value, loss_data, loss_0, loss_colloc, loss_ulb, loss_lrb = loss(
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
            y_ulb,
            t_ulb,
            x_lrb,
            y_lrb,
            t_lrb,
            weights_data,
            weights_0,
            weights_colloc,
            weights_ulb,
            weights_lrb,
        )
        grads = tape.gradient(loss_value, u_model.trainable_variables)
        grads_data = tape.gradient(loss_value, weights_data)
        grads_0 = tape.gradient(loss_value, weights_0)
        grads_colloc = tape.gradient(loss_value, weights_colloc)
        grads_ulb = tape.gradient(loss_value, weights_ulb)
        grads_lrb = tape.gradient(loss_value, weights_lrb)


    return loss_value, loss_data, loss_0, loss_colloc, loss_ulb, loss_lrb, grads, grads_data, grads_0, grads_colloc, grads_ulb, grads_lrb


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
import netCDF4
from scipy import interpolate
%matplotlib qt


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
    f_model,
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

    return (
        loss_value,
        loss_data,
        loss_0,
        loss_colloc,
        loss_ulb,
        loss_lrb,
        grads,
        grads_data,
        grads_0,
        grads_colloc,
        grads_ulb,
        grads_lrb,
    )


#%% Fit function


def fit(
    X_data_colloc_batch,
    X_0,
    X_ul,
    X_lr,
    weights_data,
    weights_0,
    weights_colloc,
    weights_ulb,
    weights_lrb,
    tf_iter,
    newton_iter,
):

    start_time = time.time()
    # create optimizers for network weights and
    tf_optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99)
    tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=0.99)
    iterator = iter(X_0)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf_optimizer, net=u_model, iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=10)
    x_0 = X_0[:,1]
    y_0 = X_0[:,2]
    t_0 = X_0[:,0]
    T_0 = X_0[:,3]

    x_ulb = X_ul[:,1]
    y_ulb = X_ul[:,2]
    t_ulb = X_ul[:,0]

    x_lrb = X_lr[:,1]
    y_lrb = X_lr[:,2]
    t_lrb = X_lr[:,0]

    # ADAM optimization
    print("Starting ADAM training")

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for epoch in range(tf_iter):
        for X_data_batch, X_colloc_batch in range(X_data_colloc_batch):

            x_data_batch = X_data_batch[:,1]
            y_data_batch = X_data_batch[:,2]
            t_data_batch = X_data_batch[:,0]
            T_data_batch = X_data_batch[:,3]

            x_colloc_batch = X_colloc_batch[:,1]
            y_colloc_batch = X_colloc_batch[:,2]
            t_colloc_batch = X_colloc_batch[:,0]
            (
                loss_value,
                loss_data,
                loss_0,
                loss_colloc,
                loss_ulb,
                loss_lrb,
                grads,
                grads_data,
                grads_0,
                grads_colloc,
                grads_ulb,
                grads_lrb,
            ) = grad(
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

            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
            tf_optimizer_weights.apply_gradients(
                zip(
                    [-grads_data, -grads_0, -grads_colloc, -grads_ulb, -grads_lrb],
                    [weights_data, weights_0, weights_colloc, weights_ulb, weights_lrb],
                )
            )

        ckpt.step.assign_add(1)
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print("It: %d, Time: %.2f" % (epoch, elapsed))
            tf.print(
                f"mse_data: {loss_data}, mse_0: {loss_0}, mse_colloc: {loss_colloc}, mse_b: {loss_ulb+loss_lrb}, Total Loss: {loss_value}"
            )
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            start_time = time.time()

    # L-BFGS optimization
    print("Starting L_BFGS training")

    loss_and_flat_grad = get_loss_and_flat_grad(
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

    lbfgs(
        loss_and_flat_grad,
        get_weights(u_model),
        Struct(),
        maxIter=newton_iter,
        learningRate=0.8,
    )


def get_loss_and_flat_grad(
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
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _, _, _, _ = loss(
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
        grad = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        return loss_value, grad_flat

    return loss_and_flat_grad


#%% Predict function


def predict(x, y, t):
    T_star = u_model(tf.concat([x, y, t], 1))
    f_T_star = f_model(x, y, t)

    return T_star.numpy(), f_T_star.numpy()

#%% Dataprep class

class dataprep:
    # Initialize class
    def __init__(self, xmax, filename="function_heat_source_out_SS_final.e"):
        nc = netCDF4.Dataset(filename)
        temps = nc.variables["vals_nod_var1"]
        temps = np.array(temps)
        self.temp = temps.reshape((110, 6, 101, 101))
        x = np.linspace(0, xmax, 101)
        y = np.linspace(0, xmax, 101)
        t = np.linspace(0, 5.994, 109)
        t = np.append(t,6)
        self.temp_func = interpolate.RegularGridInterpolator(
            (t, x, y), self.temp[:, 5, :, :]
        )

    def getCoords(self, xmin, xmax, ymin, ymax, tmin, num_x, num_y, num_t):
        # num_x, num_y: number per edge
        # num_t: number time step

        x = np.linspace(xmin, xmax, num=num_x)
        y = np.linspace(ymin, ymax, num=num_y)
        t = np.linspace(tmin, tmin+((num_t-1)*0.0555), num=num_t)
        xxx, yyy, ttt = np.meshgrid(x, y, t)
        xxx = xxx.flatten()[:, None]
        yyy = yyy.flatten()[:, None]
        ttt = ttt.flatten()[:, None]

        return np.concatenate((ttt, xxx, yyy), 1)

    def getTemps(self, coords, temp_cutoff=20000):
        temp_fromfunc = self.temp_func(coords).flatten()[:, None]
        # Get indices of those points whose temepratures are less than 3000K
        idx = np.where(temp_fromfunc <= temp_cutoff)[0]
        self.temp_upd = temp_fromfunc[idx]
        self.coords_upd = coords[idx]
        self.dataset = np.concatenate((self.coords_upd, self.temp_upd), 1)
        return self.dataset

    def minibatch(self, batchsize=128):
        rand_idx = np.random.choice(
            self.dataset.shape[0], size=batchsize, replace=False
        )
        batch = self.dataset[rand_idx, :]
        return batch

#%%
batch_size = 256
N_colloc = 200000
N_data = 200000
N_initial = 5000
N_boundary = 5000

# Dataset with ground truth
temp_data = dataprep(1)
X_data_coords = temp_data.getCoords(xmin = 0.01, xmax = 0.99, ymin = 0.01, ymax = 0.99, tmin = 0.0555, num_x = 99, num_y = 99, num_t = 88)
X_data = temp_data.getTemps(X_data_coords)
X_dataset = tf.data.Dataset.from_tensor_slices(X_data)
X_dataset_batch = X_dataset.shuffle(buffer_size=1000).take(N_data).batch(batch_size)


# Intitial Conditions dataset
temp0_data = dataprep(1)
X_0_coords = temp_data.getCoords(xmin = 0.01, xmax = 0.99, ymin = 0.01, ymax = 0.99, tmin = 0, num_x = 99, num_y = 99, num_t = 1)
X_0 = temp_data.getTemps(X_0_coords)
X_0 = tf.data.Dataset.from_tensor_slices(X_0)
X_0 = X_0.shuffle(buffer_size=1000).take(N_initial)

# Collocation Dataset
lb = np.array([0, 0, 0])
ub = np.array([4.88, 1, 1])
X_colloc = lb + (ub-lb)*lhs(3, N_colloc)
X_colloc = tf.data.Dataset.from_tensor_slices(X_colloc)
X_colloc_batch = X_colloc.shuffle(buffer_size=1000).take(N_colloc).batch(batch_size)

# Upper Lower Bundary Dataset
tempul_data = dataprep(1)
X_ul_coords = temp_data.getCoords(xmin = 0.01, xmax = 0.99, ymin = 0, ymax = 1, tmin = 0.0555, num_x = 99, num_y = 2, num_t = 88)
X_ul = temp_data.getTemps(X_ul_coords)
X_ul = tf.data.Dataset.from_tensor_slices(X_ul)
X_ul = X_ul.shuffle(buffer_size=1000).take(N_boundary)

# Left right Boundary Dataset
templr_data = dataprep(1)
X_lr_coords = temp_data.getCoords(xmin = 0, xmax = 1, ymin = 0.01, ymax = 0.99, tmin = 0.0555, num_x = 2, num_y = 99, num_t = 88)
X_lr = temp_data.getTemps(X_lr_coords)
X_lr = tf.data.Dataset.from_tensor_slices(X_lr)
X_lr = X_lr.shuffle(buffer_size=1000).take(N_boundary)

X_data_colloc_batch = tf.data.Dataset.zip((X_dataset_batch,X_colloc_batch))



# %%

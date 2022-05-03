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



#define size of the network
layer_sizes = [2, 128, 128, 128, 128, 1]

sizes_w = []
sizes_b = []

for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))


#L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0
def set_weights(model, w, sizes_w, sizes_b):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights:end_weights + sizes_b[i]]
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

#define the neural network model
def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))
    return model



#define the loss
def loss(x_f_batch, t_f_batch,
         x0, t0, u0, x_lb,
         t_lb, x_ub, t_ub,
         col_weights, u_weights):

    f_u_pred = f_model(x_f_batch, t_f_batch)
    u0_pred = u_model(tf.concat([x0, t0], 1))

    u_lb_pred, u_x_lb_pred, = u_x_model(u_model, x_lb, t_lb)
    u_ub_pred, u_x_ub_pred, = u_x_model(u_model, x_ub, t_ub)

    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))
    mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))

    mse_f_u = tf.reduce_mean(tf.square(col_weights * f_u_pred[0]))

    return  mse_0_u + mse_b_u + mse_f_u , tf.reduce_mean(tf.square((u0 - u0_pred))), mse_b_u, tf.reduce_mean(tf.square(f_u_pred))

#define the physics-based residual, we want this to be 0

@tf.function
def f_model(x,t):
    u = u_model(tf.concat([x, t],1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    c1 = tf.constant(.0001, dtype = tf.float32)
    c2 = tf.constant(5.0, dtype = tf.float32)
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u

@tf.function
def u_x_model(u_model, x, t):
    u = u_model(tf.concat([x, t],1))
    u_x = tf.gradients(u, x)
    return u, u_x

@tf.function
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_0, mse_b, mse_f = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)
        grads = tape.gradient(loss_value, u_model.trainable_variables)
        #print(grads)
        grads_col = tape.gradient(loss_value, col_weights)
        grads_u = tape.gradient(loss_value, u_weights)
        gradients_u = tape.gradient(mse_0, u_model.trainable_variables)
        gradients_f = tape.gradient(mse_f, u_model.trainable_variables)

    return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, gradients_u, gradients_f


def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, tf_iter, newton_iter):

    #Can adjust batch size for collocation points, here we set it to N_f
    batch_sz = N_f
    n_batches =  N_f // batch_sz

    start_time = time.time()
    #create optimizer s for the network weights, collocation point mask, and initial boundary mask
    tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
    tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
    tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

    print("starting Adam training")

    # For mini-batch (if used)
    for epoch in range(tf_iter):
        for i in range(n_batches):

            x0_batch = x0
            t0_batch = t0
            u0_batch = u0

            x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
            t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]
            
            loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, g_u, g_f = grad(u_model, x_f_batch, t_f_batch, x0_batch, t0_batch,  u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
            tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [col_weights, u_weights]))

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
            start_time = time.time()

    #l-bfgs-b optimization
    print("Starting L-BFGS training")

    loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

    lbfgs(loss_and_flat_grad,
      get_weights(u_model),
      Struct(), maxIter=newton_iter, learningRate=0.8)


#L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _, _ = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)
        grad = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        #print(loss_value, grad_flat)
        return loss_value, grad_flat

    return loss_and_flat_grad


def predict(X_star):
    X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
    u_star, _ = u_x_model(u_model, X_star[:,0:1],
                     X_star[:,1:2])

    f_u_star = f_model(X_star[:,0:1],
                 X_star[:,1:2])

    return u_star.numpy(), f_u_star.numpy()



# Define constants and weight vectors

lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 512
N_b = 100
N_f = 20000

col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))

#initialize the NN
u_model = neural_net(layer_sizes)

#view the NN
u_model.summary()

# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)



#grab training points from domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Grab collocation points using latin hpyercube sampling

X_f = lb + (ub-lb)*lhs(2, N_f)

x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)


X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
t0 = tf.cast(X0[:,1:2], dtype = tf.float32)

x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)

x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)


#train loop
fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, tf_iter = 10000, newton_iter = 10000)

from deepxde.backend.set_default_backend import set_default_backend
set_default_backend("tensorflow.compat.v1")

import deepxde as dde
import numpy as np

from deepxde.backend import tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random
import pickle

# set up the problem

K = 10

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    d = 1

    residual = 0

    for l in range(1, K + 1):
        residual += ((np.pi * l) ** 2 - 1) * tf.sin(l * np.pi * x[:, 0:1]) / l

    return (dy_t - d * dy_xx - tf.exp(-x[:, 1:]) * residual)

def func(x):
    residual = np.sin(np.pi * x[:, 0:1]) + np.sin(2 * np.pi * x[:, 0:1]) / 2

    for l in range(3, K + 1):
        residual += np.sin(l * np.pi * x[:, 0:1]) / l

    return (np.exp(- x[:, 1:]) * residual)


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.5)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# define the boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=12800, num_boundary=800,
    num_initial=400, solution=func, num_test=10000)


# set up the model
layer_size = [2] + [100] * 8 + [1]
activation = "sin"
initializer = "Glorot uniform"

def create_data(t):
        x = geom.uniform_points(256, True)
        t = np.full(256, t/100)
        X = np.hstack((x, t.reshape(-1, 1)))
        return X

net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=1e-5, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=100000)

y_pred = np.array([model.predict(create_data(t)) for t in range(100)])
y_sol = np.array([func(create_data(t)) for t in range(100)])

sol_data = [y_pred, y_sol]

with open(f'very_large_spectrum.pkl', 'wb') as f:
    pickle.dump(sol_data, f)
from deepxde.backend.set_default_backend import set_default_backend
set_default_backend("tensorflow.compat.v1")

import deepxde as dde
import numpy as np

from deepxde.backend import tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random
import pickle

for i in range(10, ):

    K = 1 + i/2 

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        d = 1
        return (
            dy_t
            - d * dy_xx
            - tf.math.exp(-K * x[:, 1:])
            * ((1 - K) * tf.sin(x[:, 0:1])
                + (4 - K) * tf.sin(2 * x[:, 0:1]) / 2
                + (9 - K) * tf.sin(3 * x[:, 0:1]) / 3
                + (16 - K)* tf.sin(4 * x[:, 0:1]) / 4
                + (64 - K) * tf.sin(8 * x[:, 0:1]) / 8
            )
        )

    def func(x):
        return np.exp(- K * x[:, 1:]) * (
            np.sin(x[:, 0:1])
            + np.sin(2 * x[:, 0:1]) / 2
            + np.sin(3 * x[:, 0:1]) / 3
            + np.sin(4 * x[:, 0:1]) / 4
            + np.sin(8 * x[:, 0:1]) / 8
        )


    geom = dde.geometry.Interval(-np.pi, np.pi)
    timedomain = dde.geometry.TimeDomain(0, 0.5)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain= int(K * 320),
        num_boundary= int(K * 20),
        num_initial= int(K * 10),
        solution=func,
        num_test=10000,
    )

    # set up the model
    layer_size = [2] + [50] * 6 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-5, metrics=["l2 relative error"])

    def create_data(t):
        x = geom.uniform_points(256, True)
        t = np.full(256, t/100)
        X = np.hstack((x, t.reshape(-1, 1)))
        return X

    print("Training Model with Amplitude Change: ", str(K))

    model.train(iterations=100000)

    # compare the true solution with the predicted solution
    y_pred = np.array([model.predict(create_data(t)) for t in range(100)])
    y_sol = np.array([func(create_data(t)) for t in range(100)])
    
    sol_data = [y_pred, y_sol]

    with open(f'amplitude_change_{K}.pkl', 'wb') as f:
      pickle.dump(sol_data, f)
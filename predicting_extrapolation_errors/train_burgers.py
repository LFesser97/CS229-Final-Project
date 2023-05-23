import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pickle

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = torch.device("cuda")

class BurgersEquation():

    def __init__(self):
        self._model = None
        self._geom = None

    def initialize_model(self, viscosity):
        # define the PDE
        def pde(x, y):
            dy_x = dde.grad.jacobian(y, x, i=0, j=0)
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_t + y * dy_x - viscosity * dy_xx
        
        # define a computational geometry and a time domain
        self._geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 0.5)
        geomtime = dde.geometry.GeometryXTime(self._geom, timedomain)

        # define the boundary and initial conditions
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

        # define the time PDE problem
        data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                                num_domain=25400, num_boundary=80, num_initial=160)
        
        # choose the network architecture and the training method
        net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot normal")
        self._model = dde.Model(data, net)
        self._model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1])


    def train_model(self, num_iterations=100000):
        losshistory, train_state = self._model.train(iterations=num_iterations)

    def _generate_xdata(self):
        full_domain = np.zeros(shape=(101 * 256, 2))
        for time_step in range(101):
            time = time_step * .01
            x = self._geom.uniform_points(256, True)
            t = np.full(256, time)
            x = np.hstack((x, t.reshape(-1, 1)))

            start = int(256 * time * 100)
            end = start + 256
            full_domain[start:end] = x
        
        return full_domain

    def save_preds(self, save_path):
        X = self._generate_xdata()
        model_preds = self._model.predict(X).squeeze()
        with open(save_path, 'wb') as f:
            pickle.dump(model_preds, f)


for viscosity in np.linspace(1e-3, 0.1, 50):
    viscosity = round(viscosity, 5)
    BurgersModel = BurgersEquation()
    BurgersModel.initialize_model(viscosity=viscosity)
    BurgersModel.train_model(100000)
    BurgersModel.save_preds(f"burgers_preds_{viscosity}.pkl")
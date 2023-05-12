import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
from torch import nn 
from torch import optim
from torch.autograd import grad

from burgers1d import sample_domain, sample_dirichlet_boundary

from utilities import get_model

experiment_name = "burgers_halft0.5_burgers_halft"

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else: 
#     device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def burgers1d_viscous_func(model_preds, t, x, viscosities):
    errors = []
    for i, nu in enumerate(viscosities):
        u_t = grad(outputs=model_preds[:, i], inputs=t, grad_outputs=torch.ones_like(model_preds[:, i]), create_graph=True, retain_graph=True)[0]
        u_x = grad(outputs=model_preds[:, i], inputs=x, grad_outputs=torch.ones_like(model_preds[:, i]), create_graph=True, retain_graph=True)[0]
        u_xx = grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        errors.append(u_t + model_preds[:, i].squeeze() * u_x - (nu / np.pi) * u_xx )
    return torch.vstack(errors)


def sample_domain(n=100, t_range=[0, 1]):
    sample_t = torch.zeros(size=(n,), device=device).uniform_(*t_range)
    sample_x = torch.zeros(size=(n,), device=device).uniform_(-1, 1)
    sample_t.requires_grad = True
    sample_x.requires_grad = True
    
    return sample_t, sample_x


def sample_dirichlet_boundary(n=100, ic_bc_ratio=0.8, t_range=[0, 1]):
    n_ic = int(n*ic_bc_ratio)
    n_bc = n - n_ic

    # u(t=0, x) = -sin(Pi*x)
    sample_ic_t = torch.zeros(size=(n_ic,), device=device)
    sample_ic_x = torch.zeros(size=(n_ic,), device=device).uniform_(-1, 1)
    sample_ic_input = torch.stack((sample_ic_t, sample_ic_x), dim=1)
    sample_ic_u = -torch.sin(np.pi * sample_ic_x)

    # u(t, x=+/-1) = 0
    sample_bc_t = torch.zeros(size=(n_bc,), device=device).uniform_(*t_range)
    sample_bc_x = torch.bernoulli(torch.ones(size=(n_bc,), device=device)*0.5) * 2 - 1 
    sample_bc_u = torch.zeros(size=(n_bc,), device=device)
    sample_bc_input = torch.stack((sample_bc_t, sample_bc_x), dim=1)

    return torch.vstack((sample_ic_input, sample_bc_input)), torch.cat((sample_ic_u, sample_bc_u)).unsqueeze(dim=1)


if __name__ == "__main__":
    # model = get_model(2, 1, n_hidden=8, hidden_width=150)
    viscosities = [0.01, 0.05, 0.1]
    train_t_range = [0.5, 1.0]
    model = get_model(2, len(viscosities))
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    boundary_criterion = nn.MSELoss()
    domain_criterion = nn.MSELoss()

    val_t = torch.linspace(0, 1, 100, device=device)
    val_x = torch.linspace(-1, 1, 200, device=device)
    val_tx = torch.cartesian_prod(val_t, val_x)

    train_losses = []
    domain_losses = []
    boundary_losses = []

    for _ in tqdm.trange(10_000):
        # forward pass
        domain_t, domain_x = sample_domain(n=3_000, t_range=train_t_range)
        boundary_pts, boundary_y = sample_dirichlet_boundary(n=1_000, t_range=train_t_range)

        domain_preds = model(torch.stack((domain_t, domain_x), dim=1))
        boundary_preds = model(boundary_pts)

        # backward pass
        f = burgers1d_viscous_func(domain_preds, domain_t, domain_x, viscosities)
        boundary_loss = boundary_criterion(boundary_preds, boundary_y.expand(-1, len(viscosities)))
        domain_loss = domain_criterion(f, torch.zeros_like(f))
        loss = domain_loss + boundary_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        domain_losses.append(domain_loss.item())
        boundary_losses.append(boundary_loss.item())

    plt.figure()
    plt.plot(train_losses, label="Overall", lw=1)
    plt.plot(domain_losses, label="Domain", lw=1)
    plt.plot(boundary_losses, label="Boundary", lw=1)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_loss.pdf")
    plt.yscale("log")
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_log_loss.pdf")

    with torch.no_grad(): 
        val_u = model(val_tx).squeeze().cpu().numpy()

    fig, axes = plt.subplots(len(viscosities), 1, figsize=[6, 2.5*len(viscosities)], sharex=True)
    for i, (visc, ax) in enumerate(zip(viscosities, axes.flatten())): 
        im = ax.pcolormesh(val_t, val_x, val_u[:, i].reshape(len(val_t), len(val_x)).T, shading="nearest", cmap="Spectral")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"$\\nu = {visc}$")
    plt.tight_layout()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_preds.pdf")

    fig, axes = plt.subplots(len(viscosities), 1, figsize=[4, 2*len(viscosities)], sharex=True)
    for i, (visc, ax) in enumerate(zip(viscosities, axes.flatten())): 
        ax.plot(val_x, val_u[:len(val_x), i])
        ax.set_title(f"$\\nu = {visc}$")
    plt.tight_layout()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_ic.pdf")

    
    # Transfer Learning
    last_frozen_layer = len(list(model.parameters())) - 3
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False
        if i == last_frozen_layer:
            break

    optimizer = optim.Adam(model.parameters())
    transfer_visc = 0.075
    train_losses = []
    domain_losses = []
    boundary_losses = []

    for _ in tqdm.trange(5_000):
        # forward pass
        domain_t, domain_x = sample_domain(n=3_000, t_range=[0, 0.5])
        boundary_pts, boundary_y = sample_dirichlet_boundary(n=1_000, t_range=[0, 0.5])

        domain_preds = model(torch.stack((domain_t, domain_x), dim=1))[:, [0]]
        boundary_preds = model(boundary_pts)[:, [0]]

        # backward pass
        f = burgers1d_viscous_func(domain_preds, domain_t, domain_x, [transfer_visc])
        boundary_loss = boundary_criterion(boundary_preds, boundary_y)
        domain_loss = domain_criterion(f, torch.zeros_like(f))
        loss = domain_loss + boundary_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        domain_losses.append(domain_loss.item())
        boundary_losses.append(boundary_loss.item())


    plt.figure()
    plt.plot(train_losses, label="Overall", lw=1)
    plt.plot(domain_losses, label="Domain", lw=1)
    plt.plot(boundary_losses, label="Boundary", lw=1)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_loss.pdf")
    plt.yscale("log")
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_log_loss.pdf")

    with torch.no_grad(): 
        val_u = model(val_tx).squeeze().cpu().numpy()

    plt.figure()
    im = plt.pcolormesh(val_t, val_x, val_u[:, 0].reshape(len(val_t), len(val_x)).T, shading="nearest", cmap="Spectral")
    plt.colorbar(im)
    plt.title(f"$\\nu = {transfer_visc}$")
    plt.tight_layout()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_preds.pdf")

    plt.figure()
    plt.plot(val_x, val_u[:len(val_x), 0])
    plt.title(f"$\\nu = {transfer_visc}$")
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_ic.pdf")



    # Interpolation 
    domain_t, domain_x = sample_domain(n=10_000, t_range=[0, 0.5])
    boundary_pts, boundary_y = sample_dirichlet_boundary(n=10_000, t_range=[0, 0.5])
    domain_preds = model(torch.stack((domain_t, domain_x), dim=1))[:, [0]]
    boundary_preds = model(boundary_pts)[:, [0]]

    f = burgers1d_viscous_func(domain_preds, domain_t, domain_x, [transfer_visc])
    interp_boundary_loss = boundary_criterion(boundary_preds, boundary_y)
    interp_domain_loss = domain_criterion(f, torch.zeros_like(f))
    interp_loss = interp_boundary_loss + interp_domain_loss

    # Extrapolation
    domain_t, domain_x = sample_domain(n=10_000, t_range=[0.5, 1.0])
    boundary_pts, boundary_y = sample_dirichlet_boundary(n=10_000, t_range=[0.5, 1.0])
    domain_preds = model(torch.stack((domain_t, domain_x), dim=1))[:, [0]]
    boundary_preds = model(boundary_pts)[:, [0]]

    f = burgers1d_viscous_func(domain_preds, domain_t, domain_x, [transfer_visc])
    extrap_boundary_loss = boundary_criterion(boundary_preds, boundary_y)
    extrap_domain_loss = domain_criterion(f, torch.zeros_like(f))
    extrap_loss = extrap_boundary_loss + extrap_domain_loss

    writeout_file = f"transfer_figs/{experiment_name}/burgers_transfer_burgers_losses.csv"
    if not os.path.isfile(writeout_file):
        with open(writeout_file, "w+") as f:
            f.write("interp_domain_loss,interp_boundary_loss,interp_comb_loss,extrap_domain_loss,extrap_boundary_loss,extrap_comb_loss\n")

    with open(writeout_file, "a") as f: 
        f.write(f"{interp_domain_loss},{interp_boundary_loss},{interp_loss},{extrap_domain_loss},{extrap_boundary_loss},{extrap_loss}\n")

    # save model 
    with open(writeout_file, 'r') as f:
        model_n = len(f.readlines())
    torch.save(model.state_dict(), f"transfer_figs/{experiment_name}/model_weights_{model_n}.pth")
    
        
    




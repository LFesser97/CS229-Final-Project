import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
from torch import nn 
from torch import optim
from torch.autograd import grad

from burgers1d_transfer import burgers1d_viscous_func, sample_domain, sample_dirichlet_boundary

from utilities import get_model

experiment_name = "allencahn_fullt_burgers_halft"

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else: 
#     device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allencahn_func(model_preds, t, x, ds):
    errors = []
    for i, d in enumerate(ds):
        u_t = grad(outputs=model_preds[:, i], inputs=t, grad_outputs=torch.ones_like(model_preds[:, i]), create_graph=True, retain_graph=True)[0]
        u_x = grad(outputs=model_preds[:, i], inputs=x, grad_outputs=torch.ones_like(model_preds[:, i]), create_graph=True, retain_graph=True)[0]
        u_xx = grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        errors.append(u_t - 0.0001*u_xx + d*(model_preds[:, i].squeeze()**3 - model_preds[:, i].squeeze()))
    return torch.vstack(errors)


def sample_allencahn_ic(n=100):
    # u(t=0, x) = x^2 cos(Pi*x)
    sample_ic_t = torch.zeros(size=(n,), device=device)
    sample_ic_x = torch.zeros(size=(n,), device=device).uniform_(-1, 1)
    sample_ic_input = torch.stack((sample_ic_t, sample_ic_x), dim=1)
    sample_ic_u = (sample_ic_x**2 * torch.cos(np.pi * sample_ic_x))[:, None]
    return sample_ic_input, sample_ic_u


def sample_allencahn_bc(n=100, t_range=[0, 1]):
    sample_bc_t = torch.zeros(size=(n,), device=device).uniform_(*t_range)
    sample_bc_x_left = -torch.ones_like(sample_bc_t)
    sample_bc_x_right = torch.ones_like(sample_bc_t)
    for t in [sample_bc_t, sample_bc_x_left, sample_bc_x_right]:
        t.requires_grad = True
    return sample_bc_t, sample_bc_x_left, sample_bc_x_right


def allencahn_periodic_bc(model_preds_left, model_preds_right, x_left, x_right, ds):
    neumann_errors = []
    for i in range(len(ds)): 
        u_x_left = grad(outputs=model_preds_left[:, i], inputs=x_left, grad_outputs=torch.ones_like(model_preds_left[:, i]), create_graph=True, retain_graph=True)[0]
        u_x_right = grad(outputs=model_preds_right[:, i], inputs=x_right, grad_outputs=torch.ones_like(model_preds_right[:, i]), create_graph=True, retain_graph=True)[0]
        neumann_errors.append(u_x_left - u_x_right)
    return (model_preds_left - model_preds_right) + torch.vstack(neumann_errors).T


if __name__ == "__main__":
    # model = get_model(2, 1, n_hidden=8, hidden_width=150)
    # ds = [1, 5, 10]
    ds = [1, 5, 10]
    model = get_model(2, len(ds))
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    ic_criterion = nn.MSELoss()
    bc_criterion = nn.MSELoss()
    domain_criterion = nn.MSELoss()

    val_t = torch.linspace(0, 1, 100, device=device)
    val_x = torch.linspace(-1, 1, 200, device=device)
    val_tx = torch.cartesian_prod(val_t, val_x)

    train_losses = []
    domain_losses = []
    boundary_losses = []

    for _ in tqdm.trange(10_000):
        # forward pass
        domain_t, domain_x = sample_domain(n=3_000, t_range=[0, 1])
        ic_pts, ic_y = sample_allencahn_ic(n=500)
        bc_t, bc_x_left, bc_x_right = sample_allencahn_bc(n=500, t_range=[0, 1])

        domain_preds = model(torch.stack((domain_t, domain_x), dim=1))
        ic_preds = model(ic_pts)
        bc_left_preds = model(torch.stack((bc_t, bc_x_left), dim=1))
        bc_right_preds = model(torch.stack((bc_t, bc_x_right), dim=1))

        # backward pass
        f = allencahn_func(domain_preds, domain_t, domain_x, ds)
        ic_loss = ic_criterion(ic_preds, ic_y.expand(-1, len(ds)))
        bc_errors = allencahn_periodic_bc(bc_left_preds, bc_right_preds, bc_x_left, bc_x_right, ds)
        bc_loss = bc_criterion(bc_errors, torch.zeros_like(bc_errors))
        domain_loss = domain_criterion(f, torch.zeros_like(f))
        loss = 0*domain_loss + ic_loss + 0*bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        domain_losses.append(domain_loss.item())
        boundary_losses.append(ic_loss.item() + bc_loss.item())

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

    fig, axes = plt.subplots(len(ds), 1, figsize=[6, 2.5*len(ds)], sharex=True)
    for i, (d, ax) in enumerate(zip(ds, axes.flatten())): 
        im = ax.pcolormesh(val_t, val_x, val_u[:, i].reshape(len(val_t), len(val_x)).T, shading="nearest", cmap="Spectral")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"$d = {d}$")
    plt.tight_layout()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_preds.pdf")

    fig, axes = plt.subplots(len(ds), 1, figsize=[4, 2*len(ds)], sharex=True)
    for i, (d, ax) in enumerate(zip(ds, axes.flatten())): 
        ax.plot(val_x, val_u[:len(val_x), i])
        ax.set_title(f"$d = {d}$")
    plt.tight_layout()
    plt.savefig(f"transfer_figs/{experiment_name}/transfer_train_ic.pdf")

    
    # Transfer Learning
    last_frozen_layer = len(list(model.parameters())) - 3
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False
        if i == last_frozen_layer:
            break

    optimizer = optim.Adam(model.parameters())
    transfer_visc = 0.01
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
        boundary_loss = bc_criterion(boundary_preds, boundary_y)
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
    interp_boundary_loss = bc_criterion(boundary_preds, boundary_y)
    interp_domain_loss = domain_criterion(f, torch.zeros_like(f))
    interp_loss = interp_boundary_loss + interp_domain_loss

    # Extrapolation
    domain_t, domain_x = sample_domain(n=10_000, t_range=[0.5, 1.0])
    boundary_pts, boundary_y = sample_dirichlet_boundary(n=10_000, t_range=[0.5, 1.0])
    domain_preds = model(torch.stack((domain_t, domain_x), dim=1))[:, [0]]
    boundary_preds = model(boundary_pts)[:, [0]]

    f = burgers1d_viscous_func(domain_preds, domain_t, domain_x, [transfer_visc])
    extrap_boundary_loss = bc_criterion(boundary_preds, boundary_y)
    extrap_domain_loss = domain_criterion(f, torch.zeros_like(f))
    extrap_loss = extrap_boundary_loss + extrap_domain_loss

    writeout_file = f"transfer_figs/{experiment_name}/burgers_transfer_burgers_losses.csv"
    if not os.path.isfile(writeout_file):
        with open(writeout_file, "w+") as f:
            f.write("interp_domain_loss,interp_boundary_loss,interp_comb_loss,extrap_domain_loss,extrap_boundary_loss,extrap_comb_loss\n")

    with open(writeout_file, "a") as f: 
        f.write(f"{interp_domain_loss},{interp_boundary_loss},{interp_loss},{extrap_domain_loss},{extrap_boundary_loss},{extrap_loss}\n")


    
        
    




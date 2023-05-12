import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
from torch import nn 
from torch import optim
from torch.autograd import grad

from utilities import *

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else: 
#     device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def burgers1d_viscous_func(model_preds, t, x):
    u_t = grad(outputs=model_preds, inputs=t, grad_outputs=torch.ones_like(model_preds), create_graph=True, retain_graph=True)[0]
    u_x = grad(outputs=model_preds, inputs=x, grad_outputs=torch.ones_like(model_preds), create_graph=True, retain_graph=True)[0]
    u_xx = grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    return u_t + model_preds.squeeze() * u_x - (0.01 / np.pi) * u_xx 


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
    model = get_model(2, 1)
    # model = get_model(2, 1, n_hidden=8, hidden_width=20, res=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    boundary_criterion = nn.MSELoss()
    domain_criterion = nn.MSELoss()

    train_losses = []
    domain_losses = []
    boundary_losses = [ ]
    counts = {1:0, 2:0, 3:0}

    for _ in tqdm.trange(10_000):
        domain_t, domain_x = sample_domain(n=3_000, t_range=[0, 0.5])
        boundary_pts, boundary_y = sample_dirichlet_boundary(n=1_000, t_range=[0, 0.5])

        # forward pass
        domain_preds = model(torch.stack((domain_t, domain_x), dim=1))
        boundary_preds = model(boundary_pts)

        # backward pass
        f = burgers1d_viscous_func(domain_preds, domain_t, domain_x)
        boundary_loss = boundary_criterion(boundary_preds, boundary_y)
        domain_loss = domain_criterion(f, torch.zeros_like(f))
        loss = domain_loss + boundary_loss

        # DPM
        epsilon = 0.001
        delta = 0.08
        w = 1.001

        optimizer.zero_grad()
        domain_loss.backward()
        domain_grads = get_grads(model)
        optimizer.zero_grad()
        boundary_loss.backward()
        boundary_grads = get_grads(model)
        optimizer.zero_grad()

        # print(domain_grads)
        # for g in domain_grads: 
            # print(g.shape)
        if domain_loss <= epsilon:
            set_grads(model, boundary_grads)
            counts[1] += 1       
        else: 
            combined_grads = [x + y for (x, y) in zip(domain_grads, boundary_grads)]
            if (domain_loss > epsilon) and (grad_dot(domain_grads, boundary_grads) >= 0):
                set_grads(model, combined_grads)
                counts[2] += 1       

            else: 
                set_grads(model, calc_dpm_grad(combined_grads, domain_grads, delta))
                counts[3] += 1       
        
        if domain_loss - epsilon > 0: 
            delta *= w
        else: 
            delta /= w

        optimizer.step()

        # break 
        train_losses.append(loss.item())
        domain_losses.append(domain_loss.item())
        boundary_losses.append(boundary_loss.item())


    # Interpolation 
    domain_t, domain_x = sample_domain(n=10_000, t_range=[0, 0.5])
    boundary_pts, boundary_y = sample_dirichlet_boundary(n=10_000, t_range=[0, 0.5])
    domain_preds = model(torch.stack((domain_t, domain_x), dim=1))
    boundary_preds = model(boundary_pts)

    f = burgers1d_viscous_func(domain_preds, domain_t, domain_x)
    interp_boundary_loss = boundary_criterion(boundary_preds, boundary_y)
    interp_domain_loss = domain_criterion(f, torch.zeros_like(f))
    interp_loss = interp_boundary_loss + interp_domain_loss

    # Extrapolation
    domain_t, domain_x = sample_domain(n=10_000, t_range=[0.5, 1.0])
    boundary_pts, boundary_y = sample_dirichlet_boundary(n=10_000, t_range=[0.5, 1.0])
    domain_preds = model(torch.stack((domain_t, domain_x), dim=1))
    boundary_preds = model(boundary_pts)

    f = burgers1d_viscous_func(domain_preds, domain_t, domain_x)
    extrap_boundary_loss = boundary_criterion(boundary_preds, boundary_y)
    extrap_domain_loss = domain_criterion(f, torch.zeros_like(f))
    extrap_loss = extrap_boundary_loss + extrap_domain_loss

    with open("burgers_dpm_stdarch_halftrange_losses.csv", "a") as f: 
        f.write(f"{interp_domain_loss},{interp_boundary_loss},{interp_loss},{extrap_domain_loss},{extrap_boundary_loss},{extrap_loss},{counts[1]},{counts[2]},{counts[3]}\n")


    plt.figure()
    plt.plot(train_losses, label="Overall", lw=1)
    plt.plot(domain_losses, label="Domain", lw=1)
    plt.plot(boundary_losses, label="Boundary", lw=1)
    plt.hlines(epsilon, 0, 10_000, color="k", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend()
    plt.savefig("res_halftrange_dpm_loss.pdf")
    plt.yscale("log")
    plt.savefig("res_halftrange_dpm_log_loss.pdf")

    val_t = torch.linspace(0, 1, 100, device=device)
    val_x = torch.linspace(-1, 1, 200, device=device)
    val_tx = torch.cartesian_prod(val_t, val_x)
    with torch.no_grad(): 
        val_u = model(val_tx).squeeze().cpu().numpy()

    plt.figure()
    im = plt.pcolormesh(val_t, val_x, val_u.reshape(len(val_t), len(val_x)).T, shading="nearest", cmap="Spectral")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("res_halftrange_dpm_preds.pdf")

    plt.figure()
    plt.plot(val_x, val_u[:len(val_x)])
    plt.savefig("res_halftrange_dpm_ic.pdf")

    print(counts)
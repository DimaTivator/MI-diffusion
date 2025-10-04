import torch
import numpy as np
from tqdm import tqdm, trange


def score_func(model, X_t, t, c, device="cuda"):
    p = torch.exp(-t)
    X_t = X_t.to(device)
    p = p.to(device)
    c = c.to(device)

    z = -model.denoiser(X_t, p, c).detach() / torch.sqrt(1 - p**2)
    return z


def estimate_MI_steps(model, X, dt=0.01, T=5, batch_size=100_000, device="cuda"):

    t_range = np.arange(dt / 2, T, step=dt)
    average_errors = []

    for i in trange(len(t_range) - 1):
        t, dt = torch.tensor(t_range[i]), t_range[i + 1] - t_range[i]

        idx = np.random.randint(0, len(X), batch_size)

        AB = X[idx]
        A = AB[:, : X.shape[1] // 2]
        B = AB[:, X.shape[1] // 2 :]

        z = torch.randn(size=(1, A.shape[1]))
        X_t = torch.exp(-t) * A + torch.sqrt(1 - torch.exp(-2 * t)) * z

        t = t * torch.ones((batch_size, 1))

        score_A = score_func(
            model,
            torch.cat([X_t, torch.zeros_like(X_t)], dim=1),
            t,
            torch.zeros(batch_size),
            device=device,
        ).cpu()
        
        score_A_B = score_func(
            model, torch.cat([X_t, B], dim=1), t, torch.ones(batch_size), device=device
        ).cpu()

        squared_error = torch.square(score_A - score_A_B).sum()

        err = squared_error / batch_size
        average_errors.append(err.item())

    return average_errors, np.sum(average_errors) * dt

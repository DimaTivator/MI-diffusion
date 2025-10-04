import torch
import numpy as np
from tqdm import tqdm, trange


def score_func(model, X_t, t, device="cuda"):
    p = torch.exp(-t)
    X_t = X_t.to(device)
    p = p.to(device)

    denom = torch.sqrt(1 - p**2)
    score = -model.denoiser(X_t, p).detach() / denom
    return score


def estimate_MI_monte_carlo(
    model_AB, model_A_B, X, num_iters=300_000, T=5, device="cuda"
):
    I = []

    idx = np.random.randint(0, len(X), num_iters)
    ts = torch.rand((num_iters, 1), device=device) * T
    zs = torch.randn(size=(num_iters, 1, X.shape[1])).to(device)

    for i in tqdm(range(num_iters)):
        AB = X[idx[i]].unsqueeze(0)
        AB = AB.to(device)
        t = ts[i, :]
        z = zs[i, :, :]

        X_t = torch.exp(-t) * AB + torch.sqrt(1 - torch.exp(-2 * t)) * z

        score_AB = score_func(model_AB, X_t, t).cpu()
        score_A_B = score_func(model_A_B, X_t, t).cpu()

        I.append(T * ((score_AB - score_A_B) ** 2).sum() / 2)

    return np.mean(I)


def estimate_MI_steps(model_AB, model_A_B, X, dt=0.01, T=5, batch_size=100_000):

    t_range = np.arange(2 * dt, T, step=dt)
    average_errors = []

    for i in trange(len(t_range) - 1):
        t, dt = torch.tensor(t_range[i]), t_range[i + 1] - t_range[i]

        squared_error = 0

        idx = np.random.randint(0, len(X), batch_size)

        AB = X[idx]
        z = torch.randn(size=(1, X.shape[1]))
        X_t = torch.exp(-t) * AB + torch.sqrt(1 - torch.exp(-2 * t)) * z

        t = t * torch.ones((batch_size, 1))

        score_AB = score_func(model_AB, X_t, t).cpu()
        score_A_B = score_func(model_A_B, X_t, t).cpu()

        squared_error += torch.square(score_AB - score_A_B).sum()

        err = squared_error / batch_size
        average_errors.append(err.item())

    return average_errors, np.sum(average_errors) * dt

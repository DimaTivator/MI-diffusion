import torch
from torch.utils.data import DataLoader, TensorDataset

from minde.model import *
from minde.mi import *

def estimate_MI(
    A: torch.Tensor, 
    B: torch.Tensor,
    num_epochs=100,
    batch_size_train=256,
    device='cuda',
    method='steps',  # steps / monte-carlo
    dt=0.01,
    T=5,
    batch_size_estimation=100_000,
    num_iters_estimation=100_000,
):
    dep_data = torch.cat([A, B], dim=1)
    indep_data = torch.cat([A, B[torch.randperm(B.shape[0])]], dim=1)
    
    dep_train_dataset = TensorDataset(dep_data)
    indep_train_dataset = TensorDataset(indep_data)
    
    dep_train_dataloader = DataLoader(dep_train_dataset, batch_size=batch_size_train, shuffle=True)
    indep_train_dataloader = DataLoader(indep_train_dataset, batch_size=batch_size_train, shuffle=True)
    
    model_AB = Diffusion(MLPDenoiser(dep_data.shape[1])).to(device)
    model_A_B = Diffusion(MLPDenoiser(dep_data.shape[1])).to(device)
    
    optimizer_AB = torch.optim.Adam(model_AB.parameters(), lr=1e-3)
    optimizer_A_B = torch.optim.Adam(model_A_B.parameters(), lr=1e-3)
    
    train(model_AB, optimizer_AB, dep_train_dataloader, num_epochs, device=device)
    train(model_A_B, optimizer_A_B, indep_train_dataloader, num_epochs, device=device)
    
    MI = None
    if method == 'steps':
        _, MI = estimate_MI_steps(model_AB, model_A_B, dep_data, dt, T, batch_size_estimation, device)
    elif method == 'monte-carlo':
        MI = estimate_MI_monte_carlo(model_AB, model_A_B, dep_data, num_iters_estimation, T, device)
    
    return MI
    
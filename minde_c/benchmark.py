import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from minde_c.model import *
from minde_c.mi import *


class DictDataset(Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return {"A": self.A[idx], "B": self.B[idx]}


def estimate_MI(
    A: torch.Tensor,
    B: torch.Tensor,
    num_epochs=100,
    batch_size_train=256,
    device="cuda",
    method="steps",  # steps
    dt=0.01,
    T=5,
    batch_size_estimation=100_000,
    num_iters_estimation=100_000,
):
    dep_train_dataset = DictDataset(A, B)
    dep_train_dataloader = DataLoader(
        dep_train_dataset, batch_size=batch_size_train, shuffle=True
    )

    model = Diffusion(ConditionalMLPDenoiser(A.shape[1] + B.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, dep_train_dataloader, num_epochs, device=device)

    MI = None
    if method == "steps":
        _, MI = estimate_MI_steps(model, A, B, dt, T, batch_size_estimation, device)

    return MI

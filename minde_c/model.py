import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm, trange
from IPython.display import clear_output
from functools import partial



class Diffusion(torch.nn.Module):
    def __init__(self, denoiser, loss=None, T=5):
        super().__init__()
        
        self.denoiser = denoiser
        self.loss = loss
        self.T = T
        
        if self.loss is None:
            self.loss = torch.nn.MSELoss()

    def apply_noise(self, x: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p = p.reshape((p.shape[0],) + (1,) * (len(x.shape) - 1))
        return p * x + torch.sqrt(1.0 - p**2) * z

    def get_denoising_loss(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor: 
        batch_size = A.shape[0]
        
        t = torch.rand((batch_size,), device=A.device) * self.T
        p = torch.exp(-t)
        # p = torch.sqrt(1.0 - torch.rand((batch_size,), device=x.device)**2.0)
        
        z = torch.randn((A.shape[0], A.shape[1]), device=A.device)
        
        # c == 0 -- marginal
        # c == 1 -- conditional 
        cond = int(np.random.random() > 0.5)
        c = cond * torch.ones((batch_size,), device=A.device)

        x_noisy = self.apply_noise(A, z, p)
        
        if cond == 0:
            z_pred = self.denoiser(torch.cat([x_noisy, torch.zeros_like(B).to(A.device)], dim=1), p, c)
        else:
            z_pred = self.denoiser(torch.cat([x_noisy, B], dim=1), p, c)

        return self.loss(z_pred, z)

    def sample(self, z: torch.Tensor, p_schedule: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # \overline{\alpha}_t = p_t^2
        # \overline{\alpha}_t = \prod_{s=1}^t \alpha_s
        # \alpha_s = 1 - \beta_s
        # \sigma_t^2 = \beta_t \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}

        overline_alpha = p_schedule**2
        alpha = overline_alpha[1::] / overline_alpha[:-1:]
        beta = 1.0 - alpha
        sigma = torch.sqrt(beta * (1.0 - overline_alpha[:-1:]) / (1.0 - overline_alpha[1::]))

        x = z

        for step in range(1, p_schedule.shape[0]):
            z_pred = self.denoiser(x, p_schedule[-step].repeat(x.shape[0]), c)
            
            x = (x - z_pred * beta[-step] / torch.sqrt(1.0 - overline_alpha[-step])) / torch.sqrt(alpha[-step])
            x += sigma[-step] * torch.randn(x.shape, device=x.device)

        return x

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.get_denoising_loss(A, B)


class ConditionalMLPDenoiser(torch.nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int = 256, add_dim: int = 2):
        super().__init__()

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(data_dim + add_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, data_dim // 2),
        )

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        p = p.reshape((p.shape[0],) + (1,) * (len(x.shape) - 1))
        c = c.reshape((c.shape[0],) + (1,) * (len(x.shape) - 1))

        x_p = torch.cat([x, p, c], dim=-1)

        return self.MLP(x_p)
    
    
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, dim)
        self.shift_scale = shift_scale
    
    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        if exists(t):
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t
        return x
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2 if shift_scale else dim_out)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups, shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups, shift_scale=shift_scale)
        self.lin_layer = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb
        h = self.block1(x, t=scale_shift)
        h = self.block2(h)
        return h + self.lin_layer(x)
    
    
class ConditionalUnetMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 128,
        dim_mults=(1, 1),
        resnet_block_groups: int = 8,
        add_dim: int = 2,  # corresponds to p and c
    ):
        super().__init__()
        self.add_dim = add_dim
        init_dim = default(hidden_dim, data_dim)
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_class = partial(ResnetBlock, groups=resnet_block_groups)
        self.init_lin = nn.Linear(data_dim + add_dim, init_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(add_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.ModuleList([block_class(dim_in, dim_in, time_emb_dim=time_dim)])
            )
        mid_dim = dims[-1]
        self.mid_block = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        for dim_in, dim_out in reversed(in_out):
            self.ups.append(
                nn.ModuleList([block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim)])
            )
        self.final_res_block = block_class(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_norm = nn.GroupNorm(resnet_block_groups, init_dim)
        self.final_act = nn.SiLU()
        self.final_lin = nn.Linear(init_dim, output_dim)
        # nn.init.zeros_(self.final_lin.weight)
        # nn.init.zeros_(self.final_lin.bias)
        
    def forward(self, x: torch.Tensor, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        p = p.reshape(p.shape[0], 1)
        c = c.reshape(c.shape[0], 1)
        cond = torch.cat([p, c], dim=1)
        x_input = torch.cat([x, cond], dim=1)
        x = self.init_lin(x_input)
        t = self.cond_mlp(cond)
        residual = x.clone()
        hs = []
        for (block,) in self.downs:
            x = block(x, t)
            hs.append(x)
        x = self.mid_block(x, t)
        for (block,) in self.ups:
            x = torch.cat((x, hs.pop()), dim=1)
            x = block(x, t)
        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_lin(x)
        return x


def train(model, optimizer, train_dataloader, n_epochs, device="cuda"):
    step = 0
    losses = []

    for epoch in trange(n_epochs):
        sum_loss = 0
        for batch in train_dataloader:
            if isinstance(batch, list):
                batch = batch[0]
                
            A = batch['A'].to(device)
            B = batch['B'].to(device)
            
            optimizer.zero_grad()

            loss = model(A, B)
            loss.backward()

            optimizer.step()

            sum_loss += loss.item()
            step += 1

        losses.append(sum_loss / len(train_dataloader))
        
        clear_output(wait=True)
        tqdm.write(
            f"Epoch [{epoch+1}/{n_epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
        )

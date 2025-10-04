import torch
import numpy as np
from tqdm import tqdm, trange


class Diffusion(torch.nn.Module):
    def __init__(self, denoiser, loss=None, T=5):
        super().__init__()

        self.denoiser = denoiser
        self.loss = loss
        self.T = T

        if self.loss is None:
            self.loss = torch.nn.MSELoss()

    def apply_noise(
        self, x: torch.Tensor, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        p = p.reshape((p.shape[0],) + (1,) * (len(x.shape) - 1))
        return p * x + torch.sqrt(1.0 - p**2) * z

    def get_denoising_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        t = torch.rand((batch_size,), device=x.device) * self.T
        p = torch.exp(-t)
        # p = torch.sqrt(1.0 - torch.rand((batch_size,), device=x.device)**2.0)

        z = torch.randn((x.shape[0], x.shape[1] // 2), device=x.device)

        # c == 0 -- marginal
        # c == 1 -- conditional
        cond = int(np.random.random() > 0.5)
        half = x.shape[1] // 2
        c = cond * torch.ones((batch_size,), device=x.device)

        x_noisy = self.apply_noise(x[:, :half], z, p)

        if cond == 0:
            z_pred = self.denoiser(
                torch.cat([x_noisy, torch.zeros_like(x_noisy).to(x.device)], dim=1),
                p,
                c,
            )
        else:
            z_pred = self.denoiser(torch.cat([x_noisy, x[:, half:]], dim=1), p, c)

        return self.loss(z_pred, z)

    def sample(
        self, z: torch.Tensor, p_schedule: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        # \overline{\alpha}_t = p_t^2
        # \overline{\alpha}_t = \prod_{s=1}^t \alpha_s
        # \alpha_s = 1 - \beta_s
        # \sigma_t^2 = \beta_t \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}

        overline_alpha = p_schedule**2
        alpha = overline_alpha[1::] / overline_alpha[:-1:]
        beta = 1.0 - alpha
        sigma = torch.sqrt(
            beta * (1.0 - overline_alpha[:-1:]) / (1.0 - overline_alpha[1::])
        )

        x = z

        for step in range(1, p_schedule.shape[0]):
            z_pred = self.denoiser(x, p_schedule[-step].repeat(x.shape[0]), c)

            x = (
                x - z_pred * beta[-step] / torch.sqrt(1.0 - overline_alpha[-step])
            ) / torch.sqrt(alpha[-step])
            x += sigma[-step] * torch.randn(x.shape, device=x.device)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_denoising_loss(x)


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


def train(model, optimizer, train_dataloader, n_epochs, device="cuda"):
    step = 0
    losses = []

    for epoch in trange(n_epochs):
        sum_loss = 0
        for batch in train_dataloader:
            if isinstance(batch, list):
                batch = batch[0]

            optimizer.zero_grad()

            loss = model(batch.to(device))
            loss.backward()

            optimizer.step()

            sum_loss += loss.item()
            step += 1

        losses.append(sum_loss / len(train_dataloader))

        tqdm.write(
            f"Epoch [{epoch+1}/{n_epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
        )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def noise(batch_size: int, dim):
    out = torch.empty(batch_size, dim)
    mean = torch.zeros(batch_size, dim)
    std = torch.ones(dim)
    out = torch.normal(mean, std, out=out)
    return out


def train_fn(iterations: int, loader: DataLoader, generator: nn.Module, discriminator: nn.Module, goptimizer: optim, doptimizer:optim, device: str):
    """
    Entraîne le modèle.

    Parameters:

    Returns:
    """
    generator.train()
    discriminator.train()

    dmean_loss = []
    for _ in range(iterations):
        for x in loader:
            z = noise(loader.batch_size, dim=100)
            z = z.to(device)
            f_loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(loader.batch_size), torch.zeros(loader.batch_size, device=device))
            r_loss = torch.nn.BCELoss()(discriminator(x).reshape(loader.batch_size), torch.ones(loader.batch_size, device=device))
            loss = (r_loss + f_loss) / 2
            doptimizer.zero_grad()
            loss.backward()
            doptimizer.step()
            dmean_loss.append(loss.item())
            break

    dmean_loss = sum(dmean_loss) / len(dmean_loss)

    z = noise(loader.batch_size, dim=100).to(device)
    loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(loader.batch_size), torch.ones(loader.batch_size, device=device))
    goptimizer.zero_grad()
    loss.backward()
    goptimizer.step()

    return dmean_loss, loss.item()
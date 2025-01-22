import math
import torch
import torch.nn as nn
from typing import *
from torch import Size, Tensor
from tqdm import tqdm



class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`. (i.e. epsilon)
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)

        return eps - sigma * s
    

class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """

        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt)
                    delta = tau / (eps.square().mean(dim=self.dims, keepdim=True) + 1e-6)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)

        err = (self.eps(x, t) - eps).square()
        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()

class eps_edm(nn.Module):
    r"""Converts denoising model trained in the EDM framework (Karras et al. 2022) to the epsilon used in SDA (Rozet & Louppe 2023), following the formula from appendix of Manshausen et al. 2024.

        Arguments:
            D: denoising neural network trained in EDM framework
            shape: The batch shape.
        """
    def __init__(
        self,
        D: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.D = D
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        
        
        t = t.reshape(t.shape + (1,) * len(self.shape))

        mu = self.mu(t)
        sigma = self.sigma(t)        

        return (mu / sigma) * (x / mu - self.D(x / mu, sigma / mu))

# Import necessary dependencies
from typing import Tuple
from contextlib import contextmanager

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer

# Variance-Invariance-Covariance (VIC) Regularization Loss
class VICRegularizationLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
        lambda_: float = 25.0,
        mu: float = 25.0,
        nu: float = 1.0
    ):
        super(VICRegularizationLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu

    def _off_diagonal(self, x: Tensor) -> Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, dim = z_a.shape

        # Variance loss
        std_z_a = torch.sqrt(torch.var(z_a, dim=0) + self.epsilon)
        std_z_b = torch.sqrt(torch.var(z_b, dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Invariance loss
        sim_loss = F.mse_loss(z_a, z_b)

        # Covariance loss
        z_a = z_a - torch.mean(z_a, dim=0)
        z_b = z_b - torch.mean(z_b, dim=0)
        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)
        cov_loss = (self._off_diagonal(cov_z_a).pow(2).sum() + self._off_diagonal(cov_z_b).pow(2).sum()) / dim

        # Total VIC loss
        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss

        return loss, std_loss, sim_loss, cov_loss
    
# Layer-wise Adaptive Rate Scaling (LARS) Optimizer
class LARS(Optimizer):
    def __init__(self, optimizer: Optimizer, eps: float = 1e-8, eta: float = 0.001):
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if eta < 0.0:
            raise ValueError(f"Invalid trust coefficient: {eta}")
        self.optim = optimizer
        self.eps = eps
        self.eta = eta

    def __getstate__(self):
        state = {
            'eps': self.eps,
            'eta': self.eta,
            'optim_state': self.optim.state_dict()
        }
        return state

    def __setstate__(self, state):
        self.eps = state['eps']
        self.trust_coef = state['eta']
        self.optim.load_state_dict(state['optim_state'])

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    @contextmanager
    def hide_weight_decays(self):
        # Temporarily set weight_decay to 0 in each param group and yield the original values
        weight_decays = []
        for group in self.optim.param_groups:
            wd = group.get('weight_decay', 0.0)
            weight_decays.append(wd)
            group['weight_decay'] = 0.0
        try:
            yield weight_decays
        finally:
            for group, wd in zip(self.optim.param_groups, weight_decays):
                group['weight_decay'] = wd

    def compute_adaptive_lr(self, param_norm, grad_norm, weight_decay):
        return self.eta * param_norm / (grad_norm + weight_decay * param_norm + self.eps)

    def apply_adaptive_lrs(self, weight_decays):
        # Iterate over each parameter group and scale its gradients.
        for group, weight_decay in zip(self.optim.param_groups, weight_decays):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_norm = p.data.norm(2)
                grad_norm = grad.norm(2)
                if param_norm > 0 and grad_norm > 0:
                    adaptive_lr = self.compute_adaptive_lr(param_norm, grad_norm, weight_decay)
                else:
                    adaptive_lr = 1.0
                # Apply weight decay: p.grad = p.grad + weight_decay * p.data
                p.grad.data.add_(p.data, alpha=weight_decay)
                # Scale gradient by the computed adaptive learning rate
                p.grad.data.mul_(adaptive_lr)

    def step(self, *args, **kwargs):
        # First hide weight decays, apply adaptive learning rates, then call the base optimizer
        with self.hide_weight_decays() as weight_decays:
            self.apply_adaptive_lrs(weight_decays)
            return self.optim.step(*args, **kwargs)
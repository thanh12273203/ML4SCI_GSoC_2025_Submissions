# Import necessary dependencies
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Function to compute MSE for pT prediction with an additional penalty term for overestimation
def pT_loss(pT_pred: Tensor, pT_true: Tensor, gamma: float = 1.0) -> Tensor:
    gamma = torch.tensor(gamma, dtype=pT_pred.dtype, device=pT_pred.device)
    pT_pred = torch.clamp(pT_pred, 0, None)  # ensure pT_pred is non-negative
    min_pT = pT_true.min()

    cond = pT_pred > min_pT
    loss_greater = (pT_pred - pT_true)**2 + gamma / (1.0 + torch.exp(3*(min_pT - pT_pred) - gamma))
    loss_lesser = (pT_pred - pT_true)**2 + gamma / (1.0 + torch.exp(-gamma))
    loss = torch.where(cond, loss_greater, loss_lesser)

    return loss.mean()

# Function to compute loss for eta (encourages estimations far from 0)
def eta_loss(eta_pred: Tensor, eta_true: Tensor, beta: float = 0.5) -> Tensor:
    eta_pred = torch.clamp(eta_pred, -2.5, 2.5)
    loss = (eta_pred - eta_true)**2 - beta * eta_pred**2

    return loss.mean()

# Function to compute the circular loss for phi predictions
def phi_loss(phi_pred: Tensor, phi_true: Tensor) -> Tensor:
    phi_pred = torch.clamp(phi_pred, -np.pi, np.pi)
    phi_true = torch.clamp(phi_true, -np.pi, np.pi)

    # Compute sine and cosine for both predictions and targets
    sin_pred, cos_pred = torch.sin(phi_pred), torch.cos(phi_pred)
    sin_true, cos_true = torch.sin(phi_true), torch.cos(phi_true)

    # Compute cosine similarity between predicted and true values
    cos_diff = cos_true * cos_pred + sin_true * sin_pred
    loss = 1.0 - cos_diff

    return loss.mean()

# Custom loss function for conservation of momentum
class MomentumLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, reduction: str = 'mean'):
        super(MomentumLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Extract the individual components
        pT_pred, eta_pred, phi_pred = pred[:, 0], pred[:, 1], pred[:, 2]
        pT_target, eta_target, phi_target = target[:, 0], target[:, 1], target[:, 2]
        
        # Compute momentum components for predictions
        px_pred = pT_pred * torch.cos(phi_pred)
        py_pred = pT_pred * torch.sin(phi_pred)
        pz_pred = pT_pred * torch.sinh(eta_pred)
        pred_vec = torch.stack([px_pred, py_pred, pz_pred], dim=1)  # (batch_size, 3)
        
        # Compute momentum components for targets
        px_target = pT_target * torch.cos(phi_target)
        py_target = pT_target * torch.sin(phi_target)
        pz_target = pT_target * torch.sinh(eta_target)
        target_vec = torch.stack([px_target, py_target, pz_target], dim=1)  # (batch_size, 3)
        
        # Compute cosine similarity for each sample
        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=1)
        angular_loss = 1.0 - cos_sim

        # Compute magnitude loss: MSE between the norms of the momentum vectors
        pred_norm = torch.norm(pred_vec, dim=1)
        target_norm = torch.norm(target_vec, dim=1)
        mag_loss = F.mse_loss(pred_norm, target_norm, reduction=self.reduction)
        
        # Return the weighted sum of angular and magnitude losses
        return self.alpha * angular_loss.mean() + (1 - self.alpha) * mag_loss
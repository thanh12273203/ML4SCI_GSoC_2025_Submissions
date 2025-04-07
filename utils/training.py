# Import necessary dependencies
from typing import Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from utils.custom_loss import pT_loss, eta_loss, phi_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the checkpoint
def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None
) -> Tuple[int, Dict[str, List[float]], nn.Module]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    history = checkpoint['history']
    
    return start_epoch, history, model

# Function to visualize the training progress
def plot_history(history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation metric (accuracy, for example)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_metric'], label="Train Accuracy")
    plt.plot(history['val_metric'], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Functions for CrossEntropyLoss accuracy metrics
def accuracy_metric_ce(pred: Tensor, target: Tensor) -> float:
    r"""
    Accuracy for multi-class classification (using CrossEntropyLoss).
    """
    pred_class = pred.argmax(dim=1)

    return torch.sum(pred_class == target).item() / target.size(0)

# Functions for BCEWithLogitsLoss accuracy metrics
def accuracy_metric_bce(outputs: Tensor, target: Tensor) -> float:
    """
    Computes accuracy for binary classification when using BCEWithLogitsLoss.
    Applies sigmoid on raw logits, then thresholds at 0.5.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    target = target.view(-1)
    preds = preds.view(-1)
    return torch.sum(preds == target).item() / target.size(0)

# Function for training and validation loops
def train_and_validate(
    model: Union[nn.Module, Tuple[nn.Module, nn.Module]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer, 
    metric: Optional[Callable] = None,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 20,
    sep_channels: bool = False,
    save_path: Union[str, List[str]] = 'model.pt',
    save_strategy: Optional[str] = 'val_loss',
    train_decoder: bool = False  # set to True if model is (encoder, decoder)
) -> Tuple[Dict[str, List[float]], nn.Module]:
    history = {
        'epoch': [],
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': []
    }
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    if train_decoder:
        encoder, decoder = model
        encoder.eval()
    else:
        model.train()
    
    for epoch in range(num_epochs):
        if train_decoder:
            decoder.train()
        else:
            model.train()
        epoch_loss = 0.0
        epoch_metric = 0.0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            if train_decoder:
                latent_space, _, = encoder(X)  # (batch_size, seq_len*d_model)
                outputs = decoder(latent_space)
            else:
                if sep_channels:
                    outputs = model(X[:, 0:1, :, :], X[:, 1:2, :, :])
                else:
                    outputs = model(X)
                    
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            if metric is not None:
                epoch_metric += metric(outputs, y)
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_loader)
        epoch_metric /= len(train_loader)
        
        # Validation loop
        if train_decoder:
            decoder.eval()
        else:
            model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                if train_decoder:
                    latent_space_val, _, = encoder(X_val)  # the encoder outputs a tuple
                    outputs_val = decoder(latent_space_val)
                else:
                    if sep_channels:
                        outputs_val = model(X_val[:, 0:1, :, :], X_val[:, 1:2, :, :])
                    else:
                        outputs_val = model(X_val)
                val_loss += criterion(outputs_val, y_val).item()
                if metric is not None:
                    val_metric += metric(outputs_val, y_val)
            val_loss /= len(val_loader)
            val_metric /= len(val_loader)
            
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['train_metric'].append(epoch_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, "
            f"Train Metric: {epoch_metric:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Metric: {val_metric:.4f}"
        )
        
        if scheduler is not None:
            scheduler.step()
        
        if save_strategy == 'val_loss' and val_loss < best_val_loss:
            best_val_loss = val_loss
            if train_decoder:
                torch.save(encoder.state_dict(), save_path[0])
                torch.save(decoder.state_dict(), save_path[1])
            else:
                torch.save(model.state_dict(), save_path)
        elif save_strategy == 'val_acc' and val_metric > best_val_acc:
            best_val_acc = val_metric
            if train_decoder:
                torch.save(encoder.state_dict(), save_path[0])
                torch.save(decoder.state_dict(), save_path[1])
            else:
                torch.save(model.state_dict(), save_path)
                
    if train_decoder:
        return history, decoder
    else:
        return history, model

# Function to evaluate the model on test data
def test(
    model: Union[nn.Module, Tuple[nn.Module, nn.Module]],
    test_loader: DataLoader,
    criterion: _Loss,
    metric: Optional[Callable] = None,
    sep_channels: bool = False,
    loss_type: str = 'bce',  # or 'ce'
    train_decoder: bool = False,
    model_name: str = 'model',
    classes: Optional[List[str]] = None
) -> None:
    if train_decoder:
        encoder, decoder = model
        encoder.eval()
        decoder.eval()
    else:
        model.eval()
        
    total_loss = 0.0
    total_metric = 0.0
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            if train_decoder:
                latent_space, _, = encoder(X)
                outputs = decoder(latent_space)
            else:
                if sep_channels:
                    outputs = model(X[:, 0:1, :, :], X[:, 1:2, :, :])
                else:
                    outputs = model(X)
                    
            total_loss += criterion(outputs, y).item()
            if metric is not None:
                total_metric += metric(outputs, y)
            
            # Process outputs for ROC and confusion matrix
            if loss_type == 'bce':
                # For BCE, outputs shape: (batch_size, 1)
                probs = torch.sigmoid(outputs)
                y_pred_list.append(probs.cpu().numpy())
            elif loss_type == 'ce':
                probs = F.softmax(outputs, dim=1)
                y_pred_list.append(probs.cpu().numpy())
            y_true_list.append(y.cpu().numpy())
            
    avg_loss = total_loss / len(test_loader)
    avg_metric = total_metric / len(test_loader)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    
    if loss_type == 'bce':
        y_pred_flat = y_pred.flatten()
        fpr, tpr, _ = roc_curve(y_true, y_pred_flat)
    elif loss_type == 'ce':
        # Assuming positive class is index 1.
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    
    print(f'Test Loss: {avg_loss:.4f}, Test Metric: {avg_metric:.4f}, ROC AUC: {roc_auc:.4f}')
    
    if loss_type == 'bce':
        y_pred_labels = (y_pred_flat > 0.5).astype(int)
    elif loss_type == 'ce':
        y_pred_labels = y_pred.argmax(axis=1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='orange', label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

# Function to pretrain the transformer autoencoder model
def pretrain_tmae(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 10,
    start_epoch: int = 0,
    beta: float = 0.2,  # weight to encourage estimations far from 0 in eta loss
    gamma: float = 0.2, # hyperparameter for pT loss
    history: Dict[str, List[float]] = {},
    checkpoint_path: str = 'tmae_checkpoint.pt',
    best_model_path: str = 'best_tmae_model.pt'
) -> Tuple[Dict[str, List[float]], nn.Module]:
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        best_recon_loss = min(history['train_loss'])
    else:
        best_recon_loss = float('inf')
    model.train()
    # Candidate groups for selecting the target triplet [pT, eta, phi]
    candidate_groups = [
        [0, 1, 2],     # lepton
        [5, 6, 7],     # jet 1
        [9, 10, 11],   # jet 2
        [13, 14, 15],  # jet 3
        [17, 18, 19]   # jet 4
    ]
    
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_loss_pT = 0.0
            epoch_loss_eta = 0.0
            epoch_loss_phi = 0.0
            epoch_loss_p = 0.0

            for batch in train_loader:
                # Get input data (labels are not used) and move to device
                data, _ = batch  
                data = data.to(device)  # (batch_size, 21, 1)
                batch_size, seq_len, _ = data.shape
                
                # Randomly select a candidate group for each instance
                group_choices = torch.randint(0, len(candidate_groups), (batch_size,), device=device)
                mask_tensor = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
                for i in range(batch_size):
                    group = candidate_groups[group_choices[i].item()]
                    mask_tensor[i, group] = True
                
                # Create masked input by cloning data and applying the mask
                masked_input = data.clone()
                masked_input[mask_tensor.unsqueeze(-1).expand_as(masked_input)] = 0.0
                
                # Forward pass
                _, recon = model(masked_input)

                # Build candidate groups tensor for indexing
                candidate_groups_tensor = torch.tensor(candidate_groups, device=device)  # (5, 3)
                batch_indices = torch.arange(batch_size, device=device)
                
                # For each instance, select the target triplet from the candidate group
                pT_target = data[batch_indices, candidate_groups_tensor[group_choices, 0], 0]  # (batch_size,)
                eta_target = data[batch_indices, candidate_groups_tensor[group_choices, 1], 0]  # (batch_size,)
                phi_target = data[batch_indices, candidate_groups_tensor[group_choices, 2], 0]  # (batch_size,)
                
                # Get the predictions for pT, eta, and phi
                pT_pred = recon[:, 0]
                eta_pred = recon[:, 1]
                phi_pred = recon[:, 2]
                
                # Compute individual losses (assuming you have defined pT_loss, eta_loss, phi_loss)
                loss_pT = pT_loss(pT_pred, pT_target, gamma=gamma)
                loss_eta = eta_loss(eta_pred, eta_target, beta=beta)
                loss_phi = phi_loss(phi_pred, phi_target)
                
                # Additional momentum conservation loss
                target_tensor = torch.stack([pT_target, eta_target, phi_target], dim=1)
                loss_p = criterion(recon, target_tensor)
                
                # Weighted total loss
                loss = loss_pT + loss_eta + loss_phi + loss_p
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_pT += loss_pT.item()
                epoch_loss_eta += loss_eta.item()
                epoch_loss_phi += loss_phi.item()
                epoch_loss_p += loss_p.item()
            
            epoch_loss /= len(train_loader)
            epoch_loss_pT /= len(train_loader)
            epoch_loss_eta /= len(train_loader)
            epoch_loss_phi /= len(train_loader)
            epoch_loss_p /= len(train_loader)

            # Update scheduler if provided.
            if scheduler is not None:
                scheduler.step()

            # Save the model if the current loss is the best so far
            if epoch_loss < best_recon_loss:
                best_recon_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
            
            # Save checkpoint at the end of every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            
            history['epoch'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['momentum_loss'].append(epoch_loss_p)
            history['pT_loss'].append(epoch_loss_pT)
            history['eta_loss'].append(epoch_loss_eta)
            history['phi_loss'].append(epoch_loss_phi)

            print(
                f"Pretrain Epoch [{epoch+1}/{num_epochs}], Component Losses: "
                f"(pT={epoch_loss_pT:.4f}, eta={epoch_loss_eta:.4f}, phi={epoch_loss_phi:.4f}), "
                f"Momentum Loss: {epoch_loss_p:.4f}, "
                f"Reconstruction Loss: {epoch_loss:.4f}"
            )
    except KeyboardInterrupt:
        print(f"Training paused by user at epoch {epoch}. Saving current checkpoint.")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        return history, model

    return history, model

# Function to visualize pretraining loss for the TMAE
def plot_pretrain_tmae_loss(history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(history['train_loss'], label="recon loss")
    plt.plot(history['momentum_loss'], label="momentum loss")
    plt.plot(history['pT_loss'], label="pT loss")
    plt.plot(history['eta_loss'], label="eta loss")
    plt.plot(history['phi_loss'], label="phi loss")

    # Find the epoch where the reconstruction loss is minimal
    min_epoch = history['train_loss'].index(min(history['train_loss']))
    plt.axvline(x=min_epoch, color='black', linestyle='--', label="min recon loss")

    plt.title("Pretraining Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate histograms of true vs predicted values for pT, eta, and phi
def inference_histograms(model: nn.Module, val_loader: DataLoader, seq_len: int) -> None:
    # Candidate groupse
    candidate_groups = [
        [0, 1, 2],     # lepton
        [5, 6, 7],     # jet 1
        [9, 10, 11],   # jet 2
        [13, 14, 15],  # jet 3
        [17, 18, 19]   # jet 4
    ]
    
    # Prepare lists to accumulate predictions and targets
    all_pT_pred, all_pT_true = [], []
    all_eta_pred, all_eta_true = [], []
    all_phi_pred, all_phi_true = [], []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, _ = batch  # no labels
            data = data.to(device)  # (batch_size, seq_len, 1)
            batch_size = data.size(0)
            
            # Randomly select a candidate group for each instance
            group_choices = torch.randint(0, len(candidate_groups), (batch_size,), device=device)
            mask_tensor = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            for i in range(batch_size):
                group = candidate_groups[group_choices[i].item()]
                mask_tensor[i, group] = True
            
            # Create masked input by setting the selected tokens to 0
            masked_input = data.clone()
            masked_input[mask_tensor.unsqueeze(-1).expand_as(masked_input)] = 0.0
            
            # Forward pass through the model
            _, recon = model(masked_input)  # (batch_size, 3)
            
            # Build a tensor version of candidate_groups for indexing
            candidate_groups_tensor = torch.tensor(candidate_groups, device=device)  # (5, 3)
            batch_indices = torch.arange(batch_size, device=device)
            
            # For each instance, select the target triplet using the candidate group selected
            pT_true = data[batch_indices, candidate_groups_tensor[group_choices, 0], 0]
            eta_true = data[batch_indices, candidate_groups_tensor[group_choices, 1], 0]
            phi_true = data[batch_indices, candidate_groups_tensor[group_choices, 2], 0]
            
            # Extract predictions
            pT_pred = recon[:, 0]
            eta_pred = recon[:, 1]
            phi_pred = recon[:, 2]
            
            # Append to the lists
            all_pT_pred.append(pT_pred.cpu().detach())
            all_pT_true.append(pT_true.cpu().detach())
            all_eta_pred.append(eta_pred.cpu().detach())
            all_eta_true.append(eta_true.cpu().detach())
            all_phi_pred.append(phi_pred.cpu().detach())
            all_phi_true.append(phi_true.cpu().detach())
    
    # Concatenate accumulated tensors and convert to numpy arrays for plotting
    all_pT_pred = torch.cat(all_pT_pred).numpy()
    all_pT_true = torch.cat(all_pT_true).numpy()
    all_eta_pred = torch.cat(all_eta_pred).numpy()
    all_eta_true = torch.cat(all_eta_true).numpy()
    all_phi_pred = torch.cat(all_phi_pred).numpy()
    all_phi_true = torch.cat(all_phi_true).numpy()
    
    # Create 2D histograms for each variable
    plt.figure(figsize=(18, 5))
    
    # pT histogram
    pT_min = min(all_pT_true.min(), all_pT_pred.min())
    pT_max = max(all_pT_true.max(), all_pT_pred.max())
    plt.subplot(1, 3, 1)
    plt.hist2d(all_pT_true, all_pT_pred, bins=50, cmap='gist_heat_r')
    plt.xlabel("True pT")
    plt.ylabel("Predicted pT")
    plt.title("pT Distribution")
    plt.colorbar()
    plt.plot([pT_min, pT_max], [pT_min, pT_max], color='blue', linestyle='-')
    plt.ylim(pT_min, pT_max)
    
    # Eta histogram: fixed range [-2.5, 2.5]
    plt.subplot(1, 3, 2)
    plt.hist2d(all_eta_true, all_eta_pred, bins=50, cmap='gist_heat_r', range=[[-2.5, 2.5], [-2.5, 2.5]])
    plt.xlabel("True eta")
    plt.ylabel("Predicted eta")
    plt.title("Eta Distribution")
    plt.colorbar()
    plt.plot([-2.5, 2.5], [-2.5, 2.5], color='blue', linestyle='-')
    
    # Phi histogram: fixed range [-1.75, 1.75]
    phi_min = -1.75
    phi_max = 1.75
    # # Phi histogram: fixed range [-np.pi, np.pi]
    # phi_min = -np.pi
    # phi_max = np.pi
    plt.subplot(1, 3, 3)
    plt.hist2d(all_phi_true, all_phi_pred, bins=50, cmap='gist_heat_r', range=[[phi_min, phi_max], [phi_min, phi_max]])
    plt.xlabel("True phi")
    plt.ylabel("Predicted phi")
    plt.title("Phi Distribution")
    plt.colorbar()
    plt.plot([phi_min, phi_max], [phi_min, phi_max], color='blue', linestyle='-')
    
    plt.tight_layout()
    plt.show()

# Function to pretrain the ResNet-15 with VICReg
def pretrain_VICReg(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 10,
    start_epoch: int = 0,
    history: Dict[str, List[float]] = {},
    checkpoint_path: str = 'tmae_checkpoint.pt',
    best_model_path: str = 'best_tmae_model.pt'
) -> Tuple[Dict[str, List[float]], nn.Module]:
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        best_recon_loss = min(history['train_loss'])
    else:
        best_recon_loss = float('inf')
    model.train()

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_std_loss = 0.0
            epoch_sim_loss = 0.0
            epoch_cov_loss = 0.0

            for batch in train_loader:
                x1, x2 = batch
                x1 = x1.to(device)
                x2 = x2.to(device)

                # Forward pass
                z1 = model(x1)
                z2 = model(x2)
                
                # Compute loss
                loss, std_loss, sim_loss, cov_loss = criterion(z1, z2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_std_loss += std_loss.item()
                epoch_sim_loss += sim_loss.item()
                epoch_cov_loss += cov_loss.item()
            
            epoch_loss /= len(train_loader)
            epoch_std_loss /= len(train_loader)
            epoch_sim_loss /= len(train_loader)
            epoch_cov_loss /= len(train_loader)

            # Update scheduler if provided.
            if scheduler is not None:
                scheduler.step()

            # Save the model if the current loss is the best so far
            if epoch_loss < best_recon_loss:
                best_recon_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
            
            # Save checkpoint at the end of every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            
            history['epoch'].append(epoch)
            history['VICReg_loss'].append(epoch_loss)
            history['variance_loss'].append(epoch_std_loss)
            history['invariance_loss'].append(epoch_sim_loss)
            history['covariance_loss'].append(epoch_cov_loss)

            print(
                f"Pretrain Epoch [{epoch+1}/{num_epochs}], "
                f"std_loss={epoch_std_loss:.4f}, sim_loss={epoch_sim_loss:.4f}, cov_loss={epoch_cov_loss:.4f}, "
                f"VICReg Loss: {epoch_loss:.4f}"
            )
    except KeyboardInterrupt:
        print(f"Training paused by user at epoch {epoch}. Saving current checkpoint.")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        return history, model

    return history, model

# Function to visualize pretraining loss for VICReg
def plot_pretrain_VICReg_loss(history: Dict[str, List[float]]) -> None:
    # Find the epoch where the VICReg loss is minimal
    min_epoch = history['VICReg_loss'].index(min(history['VICReg_loss']))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['variance_loss'], label="variance loss")
    plt.plot(history['invariance_loss'], label="invariance loss")
    plt.plot(history['covariance_loss'], label="covariance loss")
    plt.axvline(x=min_epoch, color='black', linestyle='--', label="min VICReg loss")
    plt.title("Pretraining Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['VICReg_loss'], color='red', label="VICReg loss")
    plt.axvline(x=min_epoch, color='black', linestyle='--', label="min VICReg loss")
    plt.title("Total VICReg Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
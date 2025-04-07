# Import necessary dependencies
from typing import Tuple

import torch
from torch import nn, Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer Masked Autoencoder (TMAE) model
class TransformerMaskedAutoencoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int, n_layers: int, dropout: float):
        super(TransformerMaskedAutoencoder, self).__init__()
        # Linear embedding
        self.linear_embedding = nn.Linear(1, d_model)
        
        # Positional embeddings (seq_len, d_model)
        self.positional_embedding = nn.Parameter(torch.randn(seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head for pT, eta, and phi
        self.recon_head = nn.Linear(seq_len*d_model, 3)
        
    def forward(self, x: Tensor) -> Tuple:
        x = self.linear_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add the positional embeddings
        x = x + self.positional_embedding
        
        # Feed through the transformer encoder
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)
        
        # Flatten the encoded output for the reconstruction head
        latent_space = torch.reshape(encoded, (encoded.size(0), -1))  # (batch_size, seq_len*d_model)

        # Reconstruction for pT, eta, and phi
        recon = self.recon_head(latent_space)  # (batch_size, 3)
        
        return latent_space, recon

# Transformer decoder for binary classification
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int, n_layers: int, dropout: float):
        super(TransformerBinaryClassifier, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Linear classification head
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # Unflatten x from (batch_size, seq_len*d_model) to (batch_size, seq_len, d_model)
        x = x.view(batch_size, self.seq_len, self.d_model)
        
        # Target shape becomes (batch_size, 1, d_model)
        tgt = self.cls_token.repeat(batch_size, 1, 1)
        
        # Pass through the transformer decoder
        decoded = self.decoder(tgt=tgt, memory=x)  # (batch_size, 1, d_model)

        # Flatten the decoded output for the classification head
        flattened_decoded = torch.reshape(decoded, (decoded.size(0), -1))  # (batch_size, d_model)

        # Final classifier
        logits = self.fc(flattened_decoded)  # (batch_size, 1)
        
        return logits.squeeze(1)  # (batch_size,) raw logits for BCEWithLogitsLoss
    
# Decoder for binary classification
class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, drop_out: float = 0.1):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        x = self.dropout(self.relu(self.linear4(x)))
        x = self.dropout(self.relu(self.linear5(x)))
        x = self.fc(x)

        return x.squeeze(-1)  # (batch_size,) raw logits for BCEWithLogitsLoss

# Example usage:
if __name__ == "__main__":
    # Run from folder ML4SCI_GSoC_2025_Submissions: python -m models.tmae
    from torch import optim
    from torch.utils.data import TensorDataset, DataLoader
    from utils import MomentumLoss, pretrain_tmae
    
    # Hyperparameters
    batch_size = 512
    d_model = 128
    seq_len = 21
    n_heads = 4
    n_layers = 6
    dropout = 0.1
    num_epochs = 10
    
    # Create the MAE model
    model = TransformerMaskedAutoencoder(
        d_model=d_model,
        seq_len=seq_len,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Dummy DataLoader: each instance is a row of 21 features
    dummy_data = torch.randn(1024, 21, 1)  # 1024 instances
    dummy_dataset = TensorDataset(dummy_data, torch.zeros(1024))
    train_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = MomentumLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Pretrain the MAE model
    model = pretrain_tmae(model, train_loader, criterion, optimizer, scheduler, num_epochs, alpha=0.1, save_path='tmae_model.pt')
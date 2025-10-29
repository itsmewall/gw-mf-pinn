# pinn.py
# Physics-Informed Neural Network (PINN) for Gravitational Wave Detection

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PINNConfig:
    # Network architecture
    INPUT_DIM: int = 5  # Your current features: snr_rms, snr_peak, crest_factor, window_sec, stride_sec
    HIDDEN_LAYERS: tuple = (128, 128, 64, 32)
    OUTPUT_DIM: int = 1  # Binary classification

    # Physics parameters
    C_SPEED_LIGHT: float = 2.998e8  # m/s
    G_GRAV_CONST: float = 6.674e-11  # m³/kg/s²
    MSUN: float = 1.989e30  # kg (solar mass)

    # Loss weights
    LAMBDA_DATA: float = 1.0
    LAMBDA_PHYSICS: float = 0.1  # Start small, tune later

    # Training
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 256
    EPOCHS: int = 100

    # Physics constraints
    M_CHIRP_MIN: float = 5.0  # Solar masses
    M_CHIRP_MAX: float = 100.0  # Solar masses
    FREQ_MIN: float = 20.0  # Hz (LIGO sensitivity lower bound)
    FREQ_MAX: float = 2048.0  # Hz

CFG = PINNConfig()

class GWPhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for gravitational wave detection.
    Enforces chirp mass frequency evolution physics in the loss function.
    """
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.INPUT_DIM
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),  # Smooth activation for automatic differentiation
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        # Classification head
        layers.append(nn.Linear(in_dim, config.OUTPUT_DIM))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        # Physics parameter estimator (latent space)
        self.physics_head = nn.Sequential(
            nn.Linear(config.HIDDEN_LAYERS[-1], 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # [M_chirp, f_peak]
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = x
        for i, layer in enumerate(self.network[:-2]):  # Stop before final layers
            features = layer(features)
        y_pred = self.network[-2:](features)
        physics_params = self.physics_head(features)
        return y_pred, physics_params

class PINNLoss(nn.Module):
    """
    Physics-informed loss for GW detection.
    Combines:
    1. Binary cross-entropy (data fitting)
    2. Physics residual (chirp mass frequency evolution)
    """
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
    def chirp_mass_physics_residual(self, M_chirp: torch.Tensor, 
                                    f_peak: torch.Tensor,
                                    snr_peak: torch.Tensor) -> torch.Tensor:
        M_chirp_kg = M_chirp * self.config.MSUN
        G = self.config.G_GRAV_CONST
        c = self.config.C_SPEED_LIGHT
        chirp_term = (np.pi * G * M_chirp_kg / (c**3)) ** (5/3)
        coeff = 96.0 / (5.0 * np.pi)
        df_dt_expected = coeff * chirp_term * (f_peak ** (11.0/3.0))
        df_dt_reference = 1e-4  # Hz/s (typical order)
        residual = torch.abs(df_dt_expected / df_dt_reference - torch.log1p(snr_peak))
        return residual
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                physics_params: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        loss_data = self.bce_loss(y_pred.squeeze(), y_true.squeeze())
        mask_positive = (y_true.squeeze() > 0.5).float()
        if mask_positive.sum() > 0:
            M_chirp = physics_params[:, 0]
            f_peak = physics_params[:, 1]
            M_chirp = torch.clamp(M_chirp, self.config.M_CHIRP_MIN, self.config.M_CHIRP_MAX)
            f_peak = torch.clamp(f_peak, self.config.FREQ_MIN, self.config.FREQ_MAX)
            snr_peak = features[:, 1]  # Index 1 = snr_peak
            residuals = self.chirp_mass_physics_residual(M_chirp, f_peak, snr_peak)
            loss_physics = (residuals * mask_positive).sum() / (mask_positive.sum() + 1e-8)
        else:
            loss_physics = torch.tensor(0.0, device=y_pred.device)
        total_loss = (self.config.LAMBDA_DATA * loss_data + 
                      self.config.LAMBDA_PHYSICS * loss_physics)
        loss_dict = {
            "total": total_loss.item(),
            "data": loss_data.item(),
            "physics": loss_physics.item()
        }
        return total_loss, loss_dict

def train_pinn(model: GWPhysicsInformedNN, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               config: PINNConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = PINNLoss(config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            y_pred, physics_params = model(batch_features)
            loss, _ = criterion(y_pred, batch_labels, physics_params, batch_features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = (y_pred.squeeze() > 0.5).float()
            train_correct += (predictions == batch_labels.squeeze()).sum().item()
            train_total += batch_labels.size(0)
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                y_pred, physics_params = model(batch_features)
                loss, _ = criterion(y_pred, batch_labels, physics_params, batch_features)
                val_loss += loss.item()
                predictions = (y_pred.squeeze() > 0.5).float()
                val_correct += (predictions == batch_labels.squeeze()).sum().item()
                val_total += batch_labels.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    return history

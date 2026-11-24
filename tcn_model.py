import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CombinedSensorAndCrossEntropyLoss(nn.Module):
    """
    Multi-Task Loss for Option B TCN:
      1. Sensor Reconstruction Loss: Unweighted MSE across 20 physical telemetry streams (channels 0..19).
      2. Anomaly Classification Loss: Weighted Cross-Entropy across 7 discrete ordinal classes (0..6).
    """

    def __init__(self, class_weights: torch.Tensor, label_loss_scale: float = 1.0):
        """
        Args:
            class_weights: 1D Tensor of shape (7,) containing rarity weights for classes 0..6.
            label_loss_scale: Multiplier scaling the classification loss relative to sensor MSE.
        """
        super(CombinedSensorAndCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights.float())
        self.label_loss_scale = label_loss_scale

    def forward(
            self,
            pred_sensors: torch.Tensor,
            pred_logits: torch.Tensor,
            target_sensors: torch.Tensor,
            target_class: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_sensors:   (B, 20) -> Predicted 20 physical telemetry signals at t+1
            pred_logits:    (B, 7)  -> Raw unnormalised classification logits for classes 0..6
            target_sensors: (B, 20) -> Ground-truth physical telemetry signals at t+1
            target_class:   (B,)    -> Ground-truth target class indices (0..6)

        Returns:
            total_loss:  Combined multi-task loss for backpropagation
            sensor_loss: Unweighted MSE on physical telemetry channels
            class_loss:  Weighted Cross-Entropy loss on discrete anomaly classes
        """
        # 1. Physical Telemetry Reconstruction Loss (MSE)
        sensor_loss = F.mse_loss(pred_sensors, target_sensors)

        # 2. Discrete Anomaly Classification Loss (Weighted Cross-Entropy)
        target_class_long = target_class.long().clamp(0, len(self.class_weights) - 1)
        class_loss = F.cross_entropy(pred_logits, target_class_long, weight=self.class_weights)

        # 3. Combined Multi-Task Loss
        total_loss = sensor_loss + (self.label_loss_scale * class_loss)
        return total_loss, sensor_loss, class_loss


class CausalConv1d(nn.Module):
    """1D Causal Convolution with Left-Padding to preserve temporal causality."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, groups: int = 1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad sequence along the time axis (dimension 2)
        return self.conv(F.pad(x, (self.padding, 0)))


class DepthwiseSeparableTCNBlock(nn.Module):
    """
    Option B Residual TCN Block:
    Uses Depthwise 1D Causal Conv + Pointwise 1x1 Conv with BatchNorm1d and SiLU Activation.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super(DepthwiseSeparableTCNBlock, self).__init__()

        # Stage 1: Depthwise 1D Conv -> Pointwise 1x1 Conv -> BatchNorm -> SiLU -> Dropout
        self.dw1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        # Stage 2: Depthwise 1D Conv -> Pointwise 1x1 Conv -> BatchNorm -> SiLU -> Dropout
        self.dw2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.bn1(self.pw1(self.dw1(x)))))
        out = self.drop2(self.act2(self.bn2(self.pw2(self.dw2(out)))))
        return out + residual


class DilatedTCNAutoencoder(nn.Module):
    """
    Option B Dilated TCN Autoencoder Architecture with Dual-Head Decoders.

    Inputs:
      - x: (B, 20, 64) -> 20 Physical Sensor signals over sequence length L=64 steps

    Outputs:
      - (pred_sensors, pred_logits):
          * pred_sensors: (B, 20) -> Forecasted telemetry at step t+1
          * pred_logits:  (B, 7)  -> Raw classification logits across 7 anomaly classes
      - f_T: (B, 128) -> Temporal Bottleneck Vector for Stage 2 Multimodal Fusion
    """

    def __init__(
            self,
            in_channels: int = 20,
            tcn_channels: int = 64,
            latent_dim: int = 128,
            num_classes: int = 7,
            num_sensors: int = 20,
            kernel_size: int = 3,
            dilations: List[int] = None,
            dropout: float = 0.2
    ):
        super(DilatedTCNAutoencoder, self).__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.input_proj = nn.Conv1d(in_channels, tcn_channels, kernel_size=1)

        # Stacked Residual TCN Blocks with exponential dilation rates
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(tcn_channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        # Temporal Bottleneck Vector (f_T) for Multimodal Fusion
        self.bottleneck = nn.Sequential(
            nn.Linear(tcn_channels, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.SiLU()
        )

        # Head 1: Continuous Physical Telemetry Decoder (20 Sensors)
        self.sensor_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_sensors)
        )

        # Head 2: Discrete Anomaly Classification Decoder (7 Logits)
        self.class_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)

        # Extract the last temporal step of the receptive field: shape (B, 64)
        h_last = out[:, :, -1]

        # Project to 128-dimensional latent temporal bottleneck vector f_T
        f_T = self.bottleneck(h_last)

        # Decode both heads in parallel
        pred_sensors = self.sensor_head(f_T)  # (B, 20)
        pred_logits = self.class_head(f_T)  # (B, 7)

        return (pred_sensors, pred_logits), f_T
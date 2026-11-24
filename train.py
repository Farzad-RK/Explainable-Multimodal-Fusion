import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from typing import Tuple

from dataset import build_cycle_bounded_windows, CycleBoundedTCNDataset, CYCLE_COLUMN, preprocess_actual_state, \
    TARGET_COLUMN
from tcn_model import DilatedTCNAutoencoder, CombinedSensorAndCrossEntropyLoss


def get_stratified_cycle_split(
        df: pd.DataFrame,
        cycle_col: str,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified Cycle Splitting by max anomaly severity per production cycle."""
    df_proc = preprocess_actual_state(df)

    cycle_summary = df_proc.groupby(cycle_col)[TARGET_COLUMN].max().reset_index()
    cycles = cycle_summary[cycle_col].values
    cycle_labels = cycle_summary[TARGET_COLUMN].values

    label_counts = pd.Series(cycle_labels).value_counts()
    singletons = label_counts[label_counts < 2].index
    n_test = int(np.ceil(len(cycles) * test_size))
    unique_classes = np.unique(cycle_labels)

    if len(unique_classes) > n_test or len(singletons) > 0:
        try:
            strat_labels = np.where(np.isin(cycle_labels, singletons), 0, cycle_labels)
            train_cycles, val_cycles = train_test_split(
                cycles, test_size=test_size, stratify=strat_labels, random_state=random_state
            )
        except Exception:
            train_cycles, val_cycles = train_test_split(
                cycles, test_size=test_size, random_state=random_state
            )
    else:
        try:
            train_cycles, val_cycles = train_test_split(
                cycles, test_size=test_size, stratify=cycle_labels, random_state=random_state
            )
        except Exception:
            train_cycles, val_cycles = train_test_split(
                cycles, test_size=test_size, random_state=random_state
            )

    return train_cycles, val_cycles


def compute_class_rarity_weights(
        y_targets: np.ndarray,
        num_classes: int = 7,
        mode: str = 'balanced'
) -> torch.Tensor:
    """Computes class rarity weights for Weighted Cross Entropy."""
    labels = np.round(y_targets[:, 20]).astype(int)
    labels = np.clip(labels, 0, num_classes - 1)

    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    smoothed_counts = np.maximum(counts, 1)

    if mode == 'balanced':
        weights = total_samples / (num_classes * smoothed_counts)
    elif mode == 'normalized':
        normal_count = smoothed_counts
        weights = normal_count / smoothed_counts
    else:
        weights = 1.0 / smoothed_counts

    weights = np.clip(weights, 0.1, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_and_evaluate(
        csv_path: str,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
        label_loss_scale: float = 1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"STARTING TCN TRAINING (WCE CLASSIFICATION + SENSOR MSE) [Device: {device}]")

    df = pd.read_csv(csv_path)
    cycle_col = CYCLE_COLUMN if CYCLE_COLUMN in df.columns else 'Q_Cell_CycleCount'

    train_cycles, val_cycles = get_stratified_cycle_split(df, cycle_col=cycle_col, test_size=0.2)

    df_train = df[df[cycle_col].isin(train_cycles)].copy()
    df_val = df[df[cycle_col].isin(val_cycles)].copy()

    x_train, y_train, states_train, scaler = build_cycle_bounded_windows(
        df_train, window_length=64, is_train=True, cycle_col=cycle_col
    )
    x_val, y_val, states_val, _ = build_cycle_bounded_windows(
        df_val, window_length=64, scaler=scaler, is_train=False, cycle_col=cycle_col
    )

    print(f"Train windows: {x_train.shape} | Val windows: {x_val.shape}")

    class_weights = compute_class_rarity_weights(y_train, num_classes=7, mode='balanced')
    print("Computed Class Rarity Weights (Classes 0..6):", np.round(class_weights.numpy(), 3))

    train_loader = DataLoader(CycleBoundedTCNDataset(x_train, y_train, states_train), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(CycleBoundedTCNDataset(x_val, y_val, states_val), batch_size=batch_size, shuffle=False)

    model = DilatedTCNAutoencoder(
        in_channels=20, tcn_channels=64, latent_dim=128, num_classes=7, num_sensors=20
    ).to(device)

    criterion = CombinedSensorAndCrossEntropyLoss(
        class_weights=class_weights.to(device),
        label_loss_scale=label_loss_scale
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_total_loss, train_sensor_loss, train_class_loss = 0.0, 0.0, 0.0

        for x_b, y_b, _ in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            target_sensors = y_b[:, :20]
            target_class = y_b[:, 20]

            optimizer.zero_grad()
            (pred_sensors, pred_logits), _ = model(x_b)

            total_loss, sensor_loss, class_loss = criterion(pred_sensors, pred_logits, target_sensors, target_class)
            total_loss.backward()
            optimizer.step()

            batch_sz = len(x_b)
            train_total_loss += total_loss.item() * batch_sz
            train_sensor_loss += sensor_loss.item() * batch_sz
            train_class_loss += class_loss.item() * batch_sz

        n_train = len(train_loader.dataset)
        train_total_loss /= n_train
        train_sensor_loss /= n_train
        train_class_loss /= n_train

        # ---------------- VALIDATION ----------------
        model.eval()
        val_total_loss, val_sensor_loss, val_label_loss = 0.0, 0.0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_b, y_b, _ in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                target_sensors = y_b[:, :20]
                target_class = y_b[:, 20]

                (pred_sensors, pred_logits), _ = model(x_b)
                total_loss, sensor_loss, class_loss = criterion(pred_sensors, pred_logits, target_sensors, target_class)

                batch_sz = len(x_b)
                val_total_loss += total_loss.item() * batch_sz
                val_sensor_loss += sensor_loss.item() * batch_sz
                val_label_loss += class_loss.item() * batch_sz

                pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()
                target_labels = target_class.long().cpu().numpy()

                all_preds.extend(pred_labels)
                all_targets.extend(target_labels)

        n_val = len(val_loader.dataset)
        val_total_loss /= n_val
        val_sensor_loss /= n_val
        val_label_loss /= n_val

        scheduler.step(val_total_loss)
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Logging
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train [Total: {train_total_loss:.4f} | Sensor MSE: {train_sensor_loss:.4f} | Label WCE: {train_class_loss:.4f}] | "
            f"Val [Total: {val_total_loss:.4f} | Sensor MSE: {val_sensor_loss:.4f} | Label WCE: {val_label_loss:.4f}] | "
            f"Weighted F1: {weighted_f1:.4f}"
        )

        # Checkpoint based ob best F1 score
        if weighted_f1 > best_val_f1:
            best_val_f1 = weighted_f1
            torch.save(model.state_dict(), "tcn_wce_encoder.pth")

    print("\n TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best validation Weighted F1: {best_val_f1:.4f}")
    print("Best encoder weights saved to: option_b_tcn_wce_encoder.pth")


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

FEATURE_COLUMNS = [
    'I_R01_Gripper_Load', 'I_R02_Gripper_Load', 'I_R03_Gripper_Load', 'I_R04_Gripper_Load',
    'I_R01_Gripper_Pot',  'I_R02_Gripper_Pot',  'I_R03_Gripper_Pot',  'I_R04_Gripper_Pot',
    'M_Conv1_Speed_mmps', 'M_Conv2_Speed_mmps',
    'M_R01_LJointAngle_Degree',
    'M_R02_SJointAngle_Degree',
    'M_R03_SJointAngle_Degree', 'M_R03_BJointAngle_Degree', 'M_R03_UJointAngle_Degree',
    'M_R04_SJointAngle_Degree', 'M_R04_LJointAngle_Degree', 'M_R04_RJointAngle_Degree',
    'M_R04_BJointAngle_Degree', 'M_R04_TJointAngle_Degree'
]

TARGET_COLUMN = 'actual_state'
CYCLE_COLUMN = 'Cycle_Count_New'
CYCLE_STATE_COLUMN = 'CycleState'

CLASS_MAPPING = {
    'Normal': 0,
    'NoNose': 1,
    'NoBody1': 2,
    'NoBody2': 3,
    'NoNose,NoBody2': 4,
    'NoBody2,NoBody1': 5,
    'NoNose,NoBody2,NoBody1': 6
}

def preprocess_actual_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust NSF-MAP target processing.
    Handles string class replacement while coercing unmapped operational states
    (such as 'E_STOPPED') safely to 0 (Normal).
    """
    df_copy = df.copy()
    if TARGET_COLUMN in df_copy.columns:
        df_copy[TARGET_COLUMN] = df_copy[TARGET_COLUMN].fillna(0)
        df_copy[TARGET_COLUMN] = df_copy[TARGET_COLUMN].replace(CLASS_MAPPING)
        # Coerce unmapped strings (e.g., 'E_STOPPED') to NaN -> fillna(0) -> int
        df_copy[TARGET_COLUMN] = (
            pd.to_numeric(df_copy[TARGET_COLUMN], errors='coerce')
            .fillna(0)
            .astype(int)
        )
    if CYCLE_STATE_COLUMN in df_copy.columns:
        df_copy[CYCLE_STATE_COLUMN] = (
            pd.to_numeric(df_copy[CYCLE_STATE_COLUMN], errors='coerce')
            .fillna(1)
            .astype(int)
        )
    return df_copy


class CycleBoundedTCNDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, y_targets: np.ndarray, cycle_states: np.ndarray):
        self.x_windows = torch.tensor(x_windows, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.float32)
        self.cycle_states = torch.tensor(cycle_states, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_windows[idx], self.y_targets[idx], self.cycle_states[idx]


def build_cycle_bounded_windows(
        df: pd.DataFrame,
        window_length: int = 64,
        scaler: Optional[StandardScaler] = None,
        is_train: bool = True,
        cycle_col: str = CYCLE_COLUMN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    df_processed = preprocess_actual_state(df)

    if cycle_col not in df_processed.columns:
        if 'Q_Cell_CycleCount' in df_processed.columns:
            cycle_col = 'Q_Cell_CycleCount'
        else:
            raise KeyError(f"Neither '{CYCLE_COLUMN}' nor 'Q_Cell_CycleCount' found in dataframe columns.")

    sensor_data = df_processed[FEATURE_COLUMNS].values
    if is_train:
        scaler = StandardScaler()
        scaled_sensors = scaler.fit_transform(sensor_data)
    else:
        if scaler is None:
            raise ValueError("StandardScaler instance must be provided when is_train=False")
        scaled_sensors = scaler.transform(sensor_data)

    scaled_df = pd.DataFrame(scaled_sensors, columns=FEATURE_COLUMNS, index=df_processed.index)
    scaled_df[TARGET_COLUMN] = df_processed[TARGET_COLUMN].values
    scaled_df[CYCLE_STATE_COLUMN] = df_processed[
        CYCLE_STATE_COLUMN].values if CYCLE_STATE_COLUMN in df_processed.columns else 1
    scaled_df[cycle_col] = df_processed[cycle_col].values

    x_list, y_list, state_list = [], [], []

    for _, group in scaled_df.groupby(cycle_col):
        group_sensors = group[FEATURE_COLUMNS].values
        group_targets = group[FEATURE_COLUMNS + [TARGET_COLUMN]].values
        group_states = group[CYCLE_STATE_COLUMN].values

        n_steps = len(group)
        if n_steps <= window_length:
            continue

        for t in range(window_length, n_steps):
            window_x = group_sensors[t - window_length: t].T
            target_y = group_targets[t]
            state_token = group_states[t]

            x_list.append(window_x)
            y_list.append(target_y)
            state_list.append(state_token)

    x_array = np.array(x_list) if len(x_list) > 0 else np.empty((0, len(FEATURE_COLUMNS), window_length))
    y_array = np.array(y_list) if len(y_list) > 0 else np.empty((0, len(FEATURE_COLUMNS) + 1))
    state_array = np.array(state_list) if len(state_list) > 0 else np.empty((0,), dtype=int)

    return x_array, y_array, state_array, scaler
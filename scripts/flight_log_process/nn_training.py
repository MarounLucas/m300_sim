import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim 
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

class DroneDataset(Dataset):
    def __init__(self, files,
                 input_mean=None,
                 target_mean=None,
                 input_std=None,
                 target_std=None):
        super().__init__()

        input_acum = []
        target_acum = []

        self.val_data = {
            'input_mean':input_mean,
            'input_std':input_std,
            'target_mean':target_mean,
            'target_std':target_std,
        }

        # Extração de Dados
        for file in files:
            df = pd.read_csv(file)

            # Entradas da Rede
            X_inputs = df[[
                'Vel_u_ms', 'Vel_v_ms','Vel_w_ms',
                'Rate_p_rads', 'Rate_q_rads','Rate_r_rads',
                'sin_Roll_deg', 'cos_Roll_deg',
                'sin_Pitch_deg', 'cos_Pitch_deg',
                'sin_Yaw_deg', 'cos_Yaw_deg',
                'Thrust_Fz_N', 'Torque_Tx_Nm', 'Torque_Ty_Nm', 'Torque_Tz_Nm'
            ]].values

            # Avlos da Rede
            Y_target = df[[ 
                'Delta_a_u', 'Delta_a_v', 'Delta_a_w', 
                'Delta_alpha_p', 'Delta_alpha_q', 'Delta_alpha_r'
            ]].values

            input_acum.append(X_inputs)
            target_acum.append(Y_target)

        # Empilhando dados
        self.raw_inputs = np.vstack(input_acum)
        self.raw_targets = np.vstack(target_acum)

        self.normalize_data()

    def normalize_data(self):
        if self.val_data['input_mean'] is None:
            self.input_mean = np.mean(self.raw_inputs, axis=0)
            self.input_std = np.std(self.raw_inputs, axis=0) + 1e-8
            self.target_mean = np.mean(self.raw_targets, axis=0)
            self.target_std = np.std(self.raw_targets, axis=0) + 1e-8
        else:
            self.input_mean = self.val_data['input_mean']
            self.input_std = self.val_data['input_std']
            self.target_mean = self.val_data['target_mean']
            self.target_std = self.val_data['target_std']

        # Noramlização
        self.norm_inputs = (self.raw_inputs - self.input_mean) / self.input_std
        self.norm_targets = (self.raw_targets - self.target_mean) / self.target_std

        # Tensores
        self.X_tensor = torch.tensor(self.norm_inputs, dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.norm_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, index):
        return self.X_tensor[index], self.Y_tensor[index]

class DroneResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.rede = nn.Sequential(
            nn.Linear(16,256),
            nn.Tanh(),

            nn.Linear(256,128),
            nn.Tanh(),

            nn.Linear(128,64),
            nn.Tanh(),

            nn.Linear(64,6)
        )
    def forward(self, x):
        return self.rede(x)

def main(train_folder_path, val_folder_path):
    # Extraindo arquivos
    training_files = glob.glob(os.path.join(train_folder_path, "*.csv"))
    val_files = glob.glob(os.path.join(val_folder_path, "*.csv"))

    # Criando Datasets
    training_dataset = DroneDataset(training_files)
    val_dataset = DroneDataset(val_files,
                               input_mean=training_dataset.input_mean,
                               input_std=training_dataset.input_std,
                               target_mean=training_dataset.target_mean,
                               target_std=training_dataset.target_std)
    
    # Criando DataLoaders
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    nn_model = DroneResidualMLP()
    otimizador = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4) #Penaliza grandes pesos

    loss_function = nn.MSELoss()
    best_val_loss = float('inf')
    best_val_loss_epoch = 0


    epocas = 100

    for epoca in range(epocas):
        nn_model.train()
        train_acum_loss = 0.0
        train_lotes = 0

        for _, (batch_x, batch_y) in enumerate(training_dataloader):
            predictions = nn_model(batch_x)
            loss = loss_function(predictions, batch_y)

            otimizador.zero_grad()
            loss.backward()

            otimizador.step()

            train_acum_loss += loss.item()
            train_lotes += 1
        train_avg_loss = train_acum_loss / train_lotes

        nn_model.eval()
        val_acum_loss = 0.0
        val_lotes = 0

        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                predictions = nn_model(batch_x)
                loss = loss_function(predictions, batch_y)

                val_acum_loss += loss.item()
                val_lotes += 1

        val_avg_loss = val_acum_loss / val_lotes

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_val_loss_epoch = epoca + 1
            torch.save(nn_model.state_dict(), 'best_m350model.pth')
            status = ' [SAVE]'

        else:
            status = ''

        print(f'Época {epoca+1:02d} / {epocas:02d} | Train Avg Loss: {train_avg_loss:.6f} | Val Avg Loss: {val_avg_loss:.6f}{status}')

    print('-'*70)
    print(f'Melhor Época: {best_val_loss_epoch} | Melhor Loss: {best_val_loss}')
        



if __name__ == "__main__":
    train_folder = '/workspace/src/m300_sim/data/flight_logs/real/ftr/train'
    val_folder = '/workspace/src/m300_sim/data/flight_logs/real/ftr/val'
    main(train_folder, val_folder)
    
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob, os

from nn_training import DroneDataset, DroneResidualMLP

def plot_graphs(Y_real, Y_pred):
    tempo = np.arange(len(Y_real)) * 0.01  # Assumindo taxa de 100Hz (0.01s por amostra)

    # 1. Gráfico das Acelerações Lineares (X, Y, Z)
    plt.figure(figsize=(12, 10))
    eixos_lin = ['X (Longitudinal)', 'Y (Lateral)', 'Z (Vertical)']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(tempo, Y_real[:, i], label='Resíduo Real (M350)', color='blue', alpha=0.7)
        plt.plot(tempo, Y_pred[:, i], label='Previsto pela Rede', color='red', linestyle='--', alpha=0.9)
        plt.ylabel(f'Acc m/s² (Eixo {eixos_lin[i]})')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.title('Validação Offline: Resíduos de Aceleração Linear (Dados Inéditos)')
        if i == 2:
            plt.xlabel('Tempo de Voo (segundos)')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('/workspace/src/m300_sim/data/nn_validation/resultado_validacao_m350_linear.png', dpi=150)
    plt.close()

    # 2. Gráfico das Acelerações Angulares (P, Q, R)
    plt.figure(figsize=(12, 10))
    eixos_ang = ['p (Roll Rate Derivative)', 'q (Pitch Rate Derivative)', 'r (Yaw Rate Derivative)']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        idx = i + 3  # Acelerações angulares ocupam as colunas 3, 4 e 5
        plt.plot(tempo, Y_real[:, idx], label='Resíduo Real (M350)', color='blue', alpha=0.7)
        plt.plot(tempo, Y_pred[:, idx], label='Previsto pela Rede', color='red', linestyle='--', alpha=0.9)
        plt.ylabel(f'Ang Acc rad/s² ({eixos_ang[i].split(" ")})')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.title('Validação Offline: Resíduos de Aceleração Angular (Dados Inéditos)')
        if i == 2:
            plt.xlabel('Tempo de Voo (segundos)')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('/workspace/src/m300_sim/data/nn_validation/resultado_validacao_m350_angular.png', dpi=150)
    plt.close()

    # 3. Painel Completo Integrado 6 DoF (3x2 Grid)
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    
    # Coluna 1: Acelerações Lineares (0, 1, 2)
    for i in range(3):
        axs[i, 0].plot(tempo, Y_real[:, i], color='blue', alpha=0.7, label='Resíduo Real (M350)')
        axs[i, 0].plot(tempo, Y_pred[:, i], color='red', linestyle='--', alpha=0.9, label='Previsto pela Rede')
        axs[i, 0].set_ylabel(f'Acc m/s² (Eixo {eixos_lin[i]})')
        axs[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axs[i, 0].set_title('Acelerações Lineares (Resíduos)')
        if i == 2:
            axs[i, 0].set_xlabel('Tempo de Voo (segundos)')
        axs[i, 0].legend(loc='upper right')

    # Coluna 2: Acelerações Angulares (3, 4, 5)
    for i in range(3):
        idx = i + 3
        axs[i, 1].plot(tempo, Y_real[:, idx], color='blue', alpha=0.7, label='Resíduo Real (M350)')
        axs[i, 1].plot(tempo, Y_pred[:, idx], color='red', linestyle='--', alpha=0.9, label='Previsto pela Rede')
        axs[i, 1].set_ylabel(f'Ang Acc rad/s² ({eixos_ang[i].split(" ")})')
        axs[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axs[i, 1].set_title('Acelerações Angulares (Resíduos)')
        if i == 2:
            axs[i, 1].set_xlabel('Tempo de Voo (segundos)')
        axs[i, 1].legend(loc='upper right')

    plt.suptitle('Gêmeo Digital DJI M350: Validação Offline de 6 DoF (Lineares e Angulares)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('/workspace/src/m300_sim/data/nn_validation/resultado_validacao_m350_6dof.png', dpi=150)
    plt.close()

def main(training_folder_path, val_folder_path):
    training_files = glob.glob(os.path.join(training_folder_path, '*.csv'))
    val_files = glob.glob(os.path.join(val_folder_path, '*.csv'))

    training_dataset = DroneDataset(training_files)
    val_dataset = DroneDataset(val_files,
                                   input_mean=training_dataset.input_mean,
                                   input_std=training_dataset.input_std,
                                   target_mean=training_dataset.target_mean,
                                   target_std=training_dataset.target_std)

    nn_model = DroneResidualMLP()
    nn_model.load_state_dict(torch.load('best_m350model.pth', weights_only=True))
    nn_model.eval()

    with torch.no_grad():
        X_val = val_dataset.X_tensor
        Y_val_norm_real = val_dataset.Y_tensor

        Y_val_norm_pred = nn_model(X_val)

    # Desnormalizar
    Y_real = (Y_val_norm_real.numpy() * val_dataset.target_std) + val_dataset.target_mean
    Y_pred = (Y_val_norm_pred.numpy() * val_dataset.target_std) + val_dataset.target_mean
    plot_graphs(Y_real,Y_pred)


if __name__ == '__main__':
    train_folder = '/workspace/src/m300_sim/data/flight_logs/real/ftr/train'
    val_folder = '/workspace/src/m300_sim/data/flight_logs/real/ftr/val'
    main(train_folder, val_folder)
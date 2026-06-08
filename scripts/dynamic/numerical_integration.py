"""
Autor: Lucas Maroun de Almeida
Descrição: Módulo para métodos de integração numérica de equações diferenciais.
"""

import numpy as np
# from numba import njit


# @njit  
def forward_euler(x_current: np.ndarray, dx: np.ndarray, dt: float) -> np.ndarray:
    """
    Realiza um passo de integração numérica utilizando o método de Euler.
    
    Args: 
        x_current (np.ndarray): Vetor de estado no instante atual (t).
        dx (np.ndarray): Derivada do estado calculada pela dinâmica.
        dt (float): Passo de integração no tempo (segundos).
    
    Returns:
        np.ndarray: Vetor de estado atualizado para o próximo instante (t + dt).
    """
    return x_current + dt * dx





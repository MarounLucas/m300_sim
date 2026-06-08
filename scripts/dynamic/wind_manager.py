"""
Descrição:
    Módulo responsável pelo gerenciamento das condições aerodinâmicas da simulação.
    Atua como um wrapper (encapsulador) para a biblioteca externa RotorPy, 
    permitindo instanciar diferentes perfis de vento (constante, rajadas de Dryden, 
    senoidal) e obter os vetores de perturbação a cada passo da integração.

Autor: Lucas Maroun de Almeida
"""

import numpy as np

from rotorpy.wind.default_winds import ConstantWind, SinusoidWind, NoWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
from rotorpy.wind.spatial_winds import WindTunnel # Adicionar no futuro


class WindManager:
    """
    Descrição:
        Gerenciador de modelos aerodinâmicos estocásticos e constantes. Converte os parâmetros 
        físicos fornecidos pelo usuário em componentes cartesianas inerciais NED e instancia o 
        perfil de vento apropriado para injetar as forças de arrasto.

    Métodos:
        __init__:
        build_profile:
        get_wind:
    """

    def __init__(self, wind_type: str = 'none', magnitude: float = 0.0, 
                 heading: float = 0.0, elevation: float = 0.0, 
                 gust_magnitude: float = 0.0) -> None:
        """
        Descrição
            Inicializa o gerenciador de vento e pré-calcula as componentes cartesianas e 
            desvios.

        Args:
            wind_type (str): Modelo de vento ('none', 'constant', 'dryden', 
                             'dryden_lp', 'sinusoid').
            magnitude (float): Velocidade média linear do vento (m/s).
            heading (float): Ângulo de direção horizontal do vento (radianos).
            elevation (float): Ângulo de elevação do vento (radianos).
            gust_magnitude (float): Magnitude máxima da rajada para modelos 
                                    estocásticos (m/s). 
        """
        self.wind_type = wind_type
        
        s_wh = np.sin(heading)
        c_wh = np.cos(heading) 
        s_we = np.sin(elevation)
        c_we = np.cos(elevation) 
        
        # Velocidade no sistema NED
        self.wx_ned = magnitude * c_wh * c_we 
        self.wy_ned = magnitude * s_wh * c_we 
        self.wz_ned = -magnitude * s_we 
        
        # Estimativa empírica de desvio padrão (sigma) para o modelo de Dryden
        # Baseado na diferença entre a intensidade máxima da rajada e a média
        sigma_total = max(0.0, (gust_magnitude - magnitude) 
                          / 3.0) if gust_magnitude > magnitude else gust_magnitude
        
        # Desvio padrão das rajadas projetado nos eixos
        self.sig_wx = abs(sigma_total * c_wh * c_we)
        self.sig_wy = abs(sigma_total * s_wh * c_we)
        self.sig_wz = abs(sigma_total * s_we)
        
        self.wind_profile = self._build_profile()

    def _build_profile(self):
        """
        Descrição:
            Avalia o tipo de vento selecionado e constrói o objeto correspondente 
            da RotorPy.

        Returns:
            Objeto base de vento que contém o método `.update()` 
        """
        if self.wind_type == 'constant':
            return ConstantWind(wx=self.wx_ned, wy=self.wy_ned, wz=self.wz_ned)
            
        elif self.wind_type == 'dryden':
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGust(avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0)
            
        elif self.wind_type == 'dryden_lp':
            avg_wind = np.array([self.wx_ned, self.wy_ned, self.wz_ned])
            sig_wind = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            return DrydenGustLP(avg_wind=avg_wind, sig_wind=sig_wind, altitude=10.0, tau=0.5)
            
        elif self.wind_type == 'sinusoid':
            amplitudes = np.array([self.sig_wx, self.sig_wy, self.sig_wz])
            frequencies = np.array([0.5, 0.5, 0.5])  # Frequência oscilatória padrão
            return SinusoidWind(amplitudes=amplitudes, frequencies=frequencies)
            
        return NoWind()

    def get_wind(self, t: float, position: np.ndarray) -> np.ndarray:
        """
        Descrição:
            Calcula e extrai o vetor de vento instantâneo para um determinado tempo e 
            posição espacial.

        Args:
            t (float): Tempo total decorrido da simulação global (segundos).
            position (np.ndarray): Vetor de posição atual da aeronave.

        Returns:
            np.ndarray: Vetor do vento instantâneo em coordenadas NED (m/s).
        """
        return self.wind_profile.update(t, position)
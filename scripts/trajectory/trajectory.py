'''
Descrição:
    Módulo responsável pelo planeamento de trajetória da aeronave. 
    Implementa a interpolação do percurso waypoints utilizando trajetórias 
    polinomiais de 5ª ordem. 

Autor: Lucas Maroun de Almeida
'''

import numpy as np

class PathManager:
    '''
    Descrição:
        Gestor de trajetória da aeronave. Recebe uma lista de coordenadas 
        (waypoints) e divide o trajeto em múltiplos segmento.
        Para cada segmento, calcula automaticamente o tempo necessário 
        com base nas velocidades e pré-calcula os coeficientes polinomiais.
    
    Métodos:
        __init__: Inicializa as listas de tempo, segmentos e coeficientes.
        calculate_segment_time: Estima o tempo necessário para percorrer um segmento.
        generate_path: Processa todos os waypoints e gera os coeficientes polinomiais.
        calculate_yaw: Calcula o ângulo de yaw desejado com base no modo selecionado.
        get_desired_state: Devolve a posição desejada num determinado instante global.
    '''
    
    def __init__(self, waypoints: list, yaw_mode: str = 'none', yaw_target: list = None,
                 xy_velocity: float = 9.2, z_velocity: float = 2.0, time_safety_factor: float = 2.6):
        '''
        Descrição:
            Inicializa o gestor de trajetória convertendo os pontos recebidos
            para vetores e preparando as estruturas de dados.

        Args:
            waypoints (list): Lista de pontos de passagem, onde cada ponto é 
                              uma lista contendo [x, y, z].
            yaw_mode (str): Modo de controle de yaw ('none', 'forward', 'target').
            yaw_target (list): Coordenada do alvo [x, y] para o modo 'target'.
            xy_velocity (float): Velocidade horizontal de cruzeiro (m/s).
            z_velocity (float): Velocidade vertical de cruzeiro (m/s).
            time_safety_factor (float): Fator de segurança para suavização da rota.
        '''
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.xy_velocity = xy_velocity
        self.z_velocity = z_velocity
        self.time_safety_factor = time_safety_factor
        
        self.segments = []
        self.segment_times = []
        self.start_times = []
        self.total_time = 0.0
        self.segment_coefficients = []

        self.yaw_mode = yaw_mode
        self.yaw_target = np.array(yaw_target, dtype=float) if yaw_target else None
        self.last_yaw = 0.0

    def calculate_segment_time(self, q0: np.ndarray, qf: np.ndarray) -> float:
        '''
        Descrição:
            Calcula o tempo estimado para o segmento definido pelas posições 
            inicial (q0) e final (qf), considerando limites de velocidade 
            independentes para os eixos XY e Z. Aplica também um fator de 
            segurança ao tempo final.

        Args:
            q0 (np.ndarray): Posição inicial do segmento [x, y, z].
            qf (np.ndarray): Posição final do segmento [x, y, z].

        Returns:
            float: Tempo estimado em segundos para concluir o segmento.
        '''
        dist_xy = np.linalg.norm(qf[0:2] - q0[0:2])
        dist_z = np.abs(qf[2] - q0[2])
        
        time_xy = dist_xy / self.xy_velocity 
        time_z = dist_z / self.z_velocity 
        
        time = max(time_xy, time_z)
        
        if time == 0:
            return 0.1 
        
        if time < 5: 
            time = 5
            
        return time * self.time_safety_factor

    def generate_path(self):
        '''
        Descrição:
            Itera sobre todos os waypoints e cria uma instância de polinómio de 
            5ª ordem para cada segmento. Calcula e armazena os coeficientes 
            polinomiais.         
        '''
        for n in range(len(self.waypoints) - 1):
            q0 = self.waypoints[n]
            qf = self.waypoints[n+1]

            t_seg = self.calculate_segment_time(q0, qf)
            self.start_times.append(self.total_time)
            self.segment_times.append(t_seg)
            self.total_time += t_seg

            traj = PolinomialTraj(q0[0:3], qf[0:3], t_seg)
            self.segments.append(traj)
            
            coefficient = traj.generate_path()
            self.segment_coefficients.append(coefficient)

        print(f"Missão Gerada: {len(self.segments)} segmentos | Tempo Total: {self.total_time:.2f}s")

    def calculate_yaw(self, idx: int, t_local: float, current_pos: np.ndarray) -> float:
        '''
        Descrição:
            Calcula o ângulo de yaw desejado com base no modo selecionado.
            'forward': Drone sempre aponta para frente;
            'target': Drone sempre aponta para um alvo;
            'none': Não há contole em yaw.
        '''
        if self.yaw_mode == 'forward':
            vel = self.segments[idx].sample_velocity(t_local)
            if np.linalg.norm(vel[0:2]) < 1e-3:
                return self.last_yaw
            
            self.last_yaw = float(np.arctan2(vel[1], vel[0]))
            return self.last_yaw

        elif self.yaw_mode == 'target':
            if self.yaw_target is None:
                raise ValueError("yaw_target não definido para o modo 'target'.")
            
            dx = self.yaw_target[0] - current_pos[0]
            dy = self.yaw_target[1] - current_pos[1]
            self.last_yaw = float(np.arctan2(dy, dx))
            return self.last_yaw

        return 0.0
    
    def get_desired_state(self, t_global: float) -> tuple:
        '''
        Descrição:
            Determina em que segmento a aeronave se encontra com base no 
            tempo global da simulação e solicita o cálculo da posição exata 
            para esse instante de tempo.

        Args:
            t_global (float): Tempo total decorrido desde o início da missão (segundos).

        Returns:
            desire_position (np.ndarray): Posição espacial desejada [x, y, z].
            desire_yaw (float): Ângulo de yaw desejado em radianos.
            idx (int): Índice do segmento de trajetória atual.
        '''
        if t_global >= self.total_time:
            final_pos = self.waypoints[-1][0:3]
            return final_pos, self.last_yaw, len(self.segments) - 1
        
        idx = 0
        for i, start_t in enumerate(self.start_times):
            if t_global >= start_t:
                idx = i
            else:
                break

        t_local = t_global - self.start_times[idx]
        desire_position = self.segments[idx].sample_position(t_local)
        desire_velocity = self.segments[idx].sample_velocity(t_local)
        desire_yaw = self.calculate_yaw(idx, t_local, desire_position)

        return desire_position, desire_velocity, desire_yaw, idx


class PolinomialTraj:
    '''
    Descrição:
        Classe responsável por construir e resolver o modelo matemático de uma 
        trajetória polinomial de 5ª ordem para um único segmento tridimensional. 
        As condições de contorno garantem que as posições são atingidas com 
        velocidade e aceleração nulas.
        
    Métodos:
        __init__: Define as condições iniciais e finais do segmento.
        generate_path: Resolve o sistema linear de restrições geométricas.
        sample_position: Calcula a posição num determinado instante de tempo.
    '''

    def __init__(self, q0: np.ndarray, qf: np.ndarray, t: float):
        '''
        Descrição:
            Inicializa o segmento polinomial estabelecendo posições iniciais 
            e finais, velocidades e acelerações a zero nos extremos.

        Args:
            q0 (np.ndarray): Vetor de posição inicial [x, y, z].
            qf (np.ndarray): Vetor de posição final [x, y, z].
            t (float): Duração total para percorrer o segmento (segundos).
        ''' 
        self.q0 = q0
        self.dq0 = np.zeros(3)
        self.ddq0 = np.zeros(3)
        self.qf = qf
        self.dqf = np.zeros(3)
        self.ddqf = np.zeros(3)
        self.time = t
        
        self.powers = np.zeros(6)
        self.vel_powers = np.zeros(6)
        self.des_position = np.zeros(3)
        self.des_velocity = np.zeros(3)
        self.x = np.zeros((6, 3))

    def generate_path(self) -> np.ndarray:
        '''
        Descrição:
            Gera o sistema de equações lineares correspondente às 6 condições 
            de contorno (posição, velocidade e aceleração no início e no fim) 
            e resolve o sistema (Ax = b) para obter a matriz de coeficientes.

            Ideia futura: escrever as equações algébricas fechadas para 
            simplificar o custo da inversão matricial.

        Returns:
            np.ndarray: Matriz (6x3) contendo os coeficientes do polinómio 
                        para as coordenadas X, Y e Z.
        '''
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, self.time, self.time**2, self.time**3, self.time**4, self.time**5],
            [0, 1, 2*self.time, 3*self.time**2, 4*self.time**3, 5*self.time**4],
            [0, 0, 2, 6*self.time, 12*self.time**2, 20*self.time**3]
        ])
        b = np.array([self.q0, self.dq0, self.ddq0, self.qf, self.dqf, self.ddqf])
        
        self.x = np.linalg.solve(A, b)
        return self.x
    
    def sample_position(self, t_atual: float) -> np.ndarray:
        '''
        Descrição:
            Avalia o polinómio de 5ª ordem num determinado instante de tempo 
            local para obter as coordenadas instantâneas da trajetória.

        Args:
            t_atual (float): Tempo decorrido desde o início do segmento (segundos).

        Returns:
            np.ndarray: Vetor contendo a posição calculada [x, y, z].
        '''
        self.powers[0] = 1.0
        self.powers[1] = t_atual
        self.powers[2] = t_atual**2
        self.powers[3] = t_atual**3
        self.powers[4] = t_atual**4
        self.powers[5] = t_atual**5
        
        np.dot(self.powers, self.x, out=self.des_position)
        return self.des_position

    def sample_velocity(self, t_atual: float) -> np.ndarray:
        '''
        Descrição:
            Avalia a primeira derivada do polinómio de 5ª ordem num determinado 
            instante de tempo local para obter as velocidades instantâneas.

        Args:
            t_atual (float): Tempo decorrido desde o início do segmento (segundos).

        Returns:
            np.ndarray: Vetor contendo a velocidade calculada [u, v, w].
        '''
        self.vel_powers[0] = 0.0
        self.vel_powers[1] = 1.0
        self.vel_powers[2] = 2.0 * t_atual
        self.vel_powers[3] = 3.0 * (t_atual**2)
        self.vel_powers[4] = 4.0 * (t_atual**3)
        self.vel_powers[5] = 5.0 * (t_atual**4)
        
        np.dot(self.vel_powers, self.x, out=self.des_velocity)
        return self.des_velocity
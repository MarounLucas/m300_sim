import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
import numpy as np

from nav_msgs.msg import Odometry
from m300_msgs.msg import FlightMode

# Importando o PathManager e o PolinomialTraj do seu módulo de trajetória
from .trajectory import PathManager, PolinomialTraj

class TrajectoryNode(Node):

    def __init__(self):
        super().__init__('trajectory_node')

        # 1. Declaração de Parâmetros da Trajetória original
        self.declare_parameter('waypoints', [0.0, 0.0, 0.0, 0.0, 0.0, -10.0])
        self.declare_parameter('yaw_mode', 'forward')

        wp_flat = self.get_parameter('waypoints').value
        yaw_mode_param = self.get_parameter('yaw_mode').value

        waypoints = [wp_flat[i:i+3] for i in range(0, len(wp_flat), 3)]
        self.path_manager = PathManager(waypoints, yaw_mode=yaw_mode_param)
        self.path_manager.generate_path()

        # 2. Configurações da FSM de Missão Unificada
        # Estados possíveis: "TRACKING", "STABILIZING", "MANUAL_MODE", "RECOVERING"
        self.mission_state = "TRACKING"
        self.traj_time = 0.0
        self.segment_idx = 0
        self.curr_position = np.zeros(3)
        self.wp_tolerance = 0.5
        self.dt = 1/100  # Executando a malha a 100Hz

        # 3. Variáveis de Controle para Alternância de Modos e Resgate
        self.current_fmd = FlightMode.AUTO
        self.paused_pos = np.zeros(3)
        self.paused_yaw = 0.0
        
        self.recovery_traj = None
        self.recovery_time = 0.0
        self.recovery_duration = 0.0

        # 4. Inicialização de Publishers e Subscribers
        self.traj_pub = self.create_publisher(Odometry, '/m300_sim/trajectory_topic', 10)
        self.state_sub = self.create_subscription(Odometry, '/m300_sim/telemetry_topic', self.pose_callback, 10)
        self.fmd_sub = self.create_subscription(FlightMode, '/m300_sim/flight_mode', self.fmd_callback, 10)

        # Timer principal da malha de trajetória
        self.create_timer(self.dt, self.traj_callback)
        self.get_logger().info('Nó de Trajetória Unificado e Inteligente Inicializado.')

    def pose_callback(self, msg: Odometry):
        """Atualiza a posição física real do drone vinda da telemetria."""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.curr_position[0:3] = px, py, pz

    def fmd_callback(self, msg: FlightMode):
        """Gerencia as transições de modo e atende às solicitações 1 e 2."""
        # [SOLICITAÇÃO 1] Transição de AUTO -> MANUAL: Armazena o último estado desejado da rota
        if self.current_fmd == FlightMode.AUTO and msg.mode == FlightMode.MANUAL:
            self.get_logger().info('Interrupção detectada: Mudança para modo MANUAL. Congelando rota.')
            self.mission_state = "MANUAL_MODE"
            # Captura exatamente onde o drone deveria estar na rota cronológica
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            
        # [SOLICITAÇÃO 2] Transição de MANUAL -> AUTO: Planeja um retorno suave por polinômio de 5ª ordem
        elif self.current_fmd == FlightMode.MANUAL and msg.mode == FlightMode.AUTO:
            self.get_logger().info('Retomando missão: Mudança para modo AUTO. Calculando retorno suave.')
            self.mission_state = "RECOVERING"
            self.recovery_time = 0.0
            
            # Garante que temos o ponto de congelamento atualizado
            self.paused_pos, _, self.paused_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            
            # Calcula o tempo necessário de voo seguro com base nas restrições de velocidade do PathManager
            self.recovery_duration = self.path_manager.calculate_segment_time(self.curr_position, self.paused_pos)
            
            # Gera dinamicamente um novo polinômio ligando a posição atual (MANUAL) ao ponto salvo da rota
            self.recovery_traj = PolinomialTraj(self.curr_position, self.paused_pos, self.recovery_duration)
            self.recovery_traj.generate_path()
            
        self.current_fmd = msg.mode

    def traj_callback(self):
        """Executa a lógica de controle temporal de acordo com o estado da FSM."""
        desire_pos = np.zeros(3)
        desire_vel = np.zeros(3)
        desire_yaw = 0.0

        # --- Estado 1: Rastreamento Normal da Rota ---
        if self.mission_state == "TRACKING":
            self.traj_time += self.dt
            desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)

            local_time = self.traj_time - self.path_manager.start_times[self.segment_idx]
            if local_time >= self.path_manager.segment_times[self.segment_idx]:
                self.get_logger().info(f'Estabilizando/Aproximando do waypoint {self.segment_idx + 1}')
                self.mission_state = "STABILIZING"
        
        # --- Estado 2: Estabilização e Verificação de Waypoints [SOLICITAÇÃO 3] ---
        elif self.mission_state == "STABILIZING":
            desire_pos, desire_vel, desire_yaw, _ = self.path_manager.get_desired_state(self.traj_time)
            target_pos = self.path_manager.waypoints[self.segment_idx + 1][0:3]
            error = np.linalg.norm(self.curr_position - target_pos)

            if error < self.wp_tolerance:
                if self.segment_idx < (len(self.path_manager.segments) - 1):
                    self.segment_idx += 1   
                    self.get_logger().info(f'Prosseguindo para o waypoint {self.segment_idx + 2}')
                    self.mission_state = "TRACKING"
                else:
                    # [SOLICITAÇÃO 3] Chegou ao fim da rota: Permanece em hover fixo indefinidamente
                    # O tempo de trajetória não avança mais e a velocidade desejada enviada será zero.
                    pass

        # --- Estado 3: Modo Manual Ativo (Voo Livre por Joystick) ---
        elif self.mission_state == "MANUAL_MODE":
            # Enquanto o piloto controla, publicamos a posição física atual como referência suave 
            # para evitar sobressaltos nas malhas internas do CascadeController
            desire_pos = self.curr_position.copy()
            desire_vel = np.zeros(3)
            desire_yaw = self.paused_yaw

        # --- Estado 4: Recuperação Suave de Trajetória [SOLICITAÇÃO 2] ---
        elif self.mission_state == "RECOVERING":
            self.recovery_time += self.dt
            
            # Amostra o polinômio de transição calculado
            desire_pos = self.recovery_traj.sample_position(self.recovery_time)
            desire_vel = self.recovery_traj.sample_velocity(self.recovery_time)
            desire_yaw = self.paused_yaw

            # Quando a curva de transição chega ao fim, retorna à rota cronológica original
            if self.recovery_time >= self.recovery_duration:
                self.get_logger().info('Drone posicionado de volta na rota com sucesso. Reiniciando missão.')
                if self.traj_time >= self.path_manager.total_time:
                    self.mission_state = "STABILIZING"
                else:
                    self.mission_state = "TRACKING"

        # --- Publicação da Referência de Odometria ---
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = float(desire_pos[0])
        msg.pose.pose.position.y = float(desire_pos[1])
        msg.pose.pose.position.z = float(desire_pos[2])

        rot = Rotation.from_euler('xyz', [0.0, 0.0, desire_yaw])
        qx, qy, qz, qw = rot.as_quat()    
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(desire_vel[0])
        msg.twist.twist.linear.y = float(desire_vel[1])
        msg.twist.twist.linear.z = float(desire_vel[2])

        self.traj_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    vec_pub_node = TrajectoryNode()
    try:
        rclpy.spin(vec_pub_node)
    except KeyboardInterrupt:
        pass
    finally:
        vec_pub_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
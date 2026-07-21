from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

# Importa as classes que criamos nos outros arquivos
from .tab_aircraft import TabAircraft
from .tab_mission import TabMission
from .tab_telemetry import TabTelemetry  

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador M350")
        self.resize(1920, 1080)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        
        # Instancia as abas modulares
        self.tab_aircraft = TabAircraft()
        self.tab_inputs = TabMission(self)
        self.tab_outputs = TabTelemetry(self) 

        self.tabs.addTab(self.tab_aircraft, "Aircraft Parameters")
        self.tabs.addTab(self.tab_inputs, "Mission Configuration")
        self.tabs.addTab(self.tab_outputs, "Results and Telemetry")
        
        main_layout.addWidget(self.tabs)
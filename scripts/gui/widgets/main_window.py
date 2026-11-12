"""Main window module for the M350 Simulator GUI."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .tab_aircraft import TabAircraft
from .tab_mission import TabMission
from .tab_telemetry import TabTelemetry


class MainWindow(QMainWindow):
    """Assemble the aircraft, mission, and telemetry tabs in the main window.

    Attributes:
        tabs: Tab container holding all simulator panels.
        tab_aircraft: Tab for configuring aircraft physical parameters.
        tab_inputs: Tab for configuring mission waypoints and wind profiles.
        tab_outputs: Tab for viewing real-time telemetry and flight logs.
    """

    def __init__(self) -> None:
        """Initialize the main window layout and modular tabs."""
        super().__init__()
        self.setWindowTitle('M350 Simulator')
        self.resize(1920, 1080)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()

        self.tab_aircraft = TabAircraft()
        self.tab_inputs = TabMission(self)
        self.tab_outputs = TabTelemetry(self)

        self.tabs.addTab(self.tab_aircraft, 'Aircraft Parameters')
        self.tabs.addTab(self.tab_inputs, 'Mission Configuration')
        self.tabs.addTab(self.tab_outputs, 'Results and Telemetry')

        main_layout.addWidget(self.tabs)

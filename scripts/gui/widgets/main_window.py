"""Main window module for the M350 Simulator GUI.

This module defines the primary application window, assembling various
modular tabs (Aircraft, Mission, Telemetry) into a unified interface.
"""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .tab_aircraft import TabAircraft
from .tab_mission import TabMission
from .tab_telemetry import TabTelemetry


class MainWindow(QMainWindow):
    """The main application window for the simulator.

    Assembles the graphical user interface by instantiating and arranging
    the modular tab widgets within a central layout.

    Attributes:
        tabs (QTabWidget): The main tab container holding all module panels.
        tab_aircraft (TabAircraft): Tab for configuring aircraft parameters.
        tab_inputs (TabMission): Tab for configuring mission parameters.
        tab_outputs (TabTelemetry): Tab for viewing telemetry and results.
    """

    def __init__(self) -> None:
        """Initializes the main window, layout, and modular tabs."""
        super().__init__()
        self.setWindowTitle("M350 Simulator")
        self.resize(1920, 1080)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()

        self.tab_aircraft = TabAircraft()
        self.tab_inputs = TabMission(self)
        self.tab_outputs = TabTelemetry(self)

        self.tabs.addTab(self.tab_aircraft, "Aircraft Parameters")
        self.tabs.addTab(self.tab_inputs, "Mission Configuration")
        self.tabs.addTab(self.tab_outputs, "Results and Telemetry")

        main_layout.addWidget(self.tabs)
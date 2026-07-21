"""Mission configuration UI module for the UAV simulator.

This module provides the graphical interface for setting waypoints, trajectory
planning parameters, and environmental conditions (including live OpenWeather 
data integration). It exports these configurations to ROS 2 YAML files.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import requests
import yaml
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# --- MATPLOTLIB MATHTEXT CONFIGURATION ---
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["text.color"] = "white"
matplotlib.rcParams["axes.labelcolor"] = "white"
matplotlib.rcParams["xtick.color"] = "#aaaaaa"
matplotlib.rcParams["ytick.color"] = "#aaaaaa"


# =========================================================================
# PATH CONFIGURATION
# =========================================================================
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent.parent

if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from paths import MISSIONS_DIR, ROS_CONFIG_DIR, ROS_INSTALL_DIR


# =========================================================================
# AUXILIARY CLASSES (HELP DIALOG AND CUSTOM MSGBOX)
# =========================================================================
class MissionHelpDialog(QDialog):
    """Floating help dialog detailing mission configuration parameters."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the mission help dialog and its HTML content.

        Args:
            parent (Optional[QWidget]): The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Documentation - Mission Configuration")
        self.resize(850, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        html_content = (
            "<html><head><style>"
            "body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; "
            "line-height: 1.6; color: #e0e0e0; padding: 15px 25px; }"
            "h2 { color: #00d4ff; margin-top: 10px; border-bottom: 1px solid #555; "
            "padding-bottom: 5px;}"
            "h3 { color: #4caf50; margin-top: 25px; margin-bottom: 5px; }"
            "ul { margin-top: 5px; padding-left: 25px; }"
            "li { margin-bottom: 8px; }"
            "b { color: #ffffff; }"
            "code { background-color: #444; padding: 2px 4px; border-radius: 3px; "
            "color: #ffcc00;}"
            "</style></head><body>"
            "<h2>Mission Configuration Guide</h2>"
            "<p>This section allows you to define the UAV's flight path, trajectory "
            "parameters, and environmental conditions for the integration algorithms.</p>"
            "<h3>1. Waypoints Configuration</h3><ul>"
            "<li><b>Coordinate System:</b> Choose between Cartesian (X, Y, Z in meters) "
            "or Geodesic (Latitude, Longitude, Altitude).</li>"
            "<li><b>Adding Points:</b> Manually input coordinates or load a batch from a "
            "<code>.csv</code> or <code>.json</code> file.</li>"
            "<li><b>Table Tools:</b> Select rows to Edit, Duplicate, Reorder, or Remove "
            "waypoints dynamically.</li></ul>"
            "<h3>2. Trajectory Planning</h3><ul>"
            "<li><b>Generation Type:</b> Mathematical method to interpolate waypoints "
            "(e.g., 5th Degree Polynomial for smooth, jerk-free motion profiles).</li>"
            "<li><b>Max Speed (XY) [m/s]:</b> Maximum horizontal speed allowed.</li>"
            "<li><b>Max Speed (Z) [m/s]:</b> Maximum vertical speed allowed.</li>"
            "<li><b>Yaw Mode:</b> Defines the drone's heading profile: <br>"
            "- <i>Free:</i> keeps the initial yaw.<br>"
            "- <i>Forward:</i> continuously points the nose to the velocity vector.<br>"
            "- <i>Target:</i> nose always points to a specific XYZ coordinate lock.</li></ul>"
            "<h3>3. Environment & Location Parameters</h3><ul>"
            "<li><b>Wind Type:</b> Simulation wind disturbance model.</li>"
            "<li><b>Base Magnitude [m/s]:</b> Base persistent wind speed.</li>"
            "<li><b>Heading / Elevation [&deg;]:</b> Wind direction vector angles.</li>"
            "<li><b>Max Gust [m/s]:</b> Maximum magnitude added dynamically by gusts.</li>"
            "<li><b>OpenWeather Data:</b> Connects to real-time meteorological servers "
            "based on global coordinates to fetch live wind speeds, heading and gusts!</li>"
            "</ul></body></html>"
        )
        browser.setHtml(html_content)
        layout.addWidget(browser)


class CustomMessageBox(QDialog):
    """Custom dialog box replacing QMessageBox for perfect responsiveness."""

    def __init__(
        self,
        title: str,
        main_text: str,
        detail_text: str = "",
        msg_type: str = "info",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initializes the custom message box with tailored styling.

        Args:
            title (str): The window title.
            main_text (str): The primary message header.
            detail_text (str): Optional secondary explanatory text.
            msg_type (str): Type of prompt ('info', 'question', 'success', 'error').
            parent (Optional[QWidget]): The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; }\n"
            "QLabel { color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; "
            "font-size: 14px;}\n"
            "QPushButton { background-color: #0d6efd; color: white; border-radius: 4px; "
            "padding: 6px 16px; font-weight: bold; min-width: 80px; }\n"
            "QPushButton:hover { background-color: #0b5ed7; }"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        lbl_main = QLabel(f"<h3 style='margin: 0;'>{main_text}</h3>")
        lbl_main.setWordWrap(True)
        layout.addWidget(lbl_main)

        if detail_text:
            lbl_detail = QLabel(detail_text)
            lbl_detail.setWordWrap(True)
            lbl_detail.setStyleSheet("color: #a0a0a0; font-size: 13px;")
            layout.addWidget(lbl_detail)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.result = QMessageBox.StandardButton.No

        if msg_type == "question":
            btn_yes = QPushButton("Yes")
            btn_yes.clicked.connect(self.accept_yes)
            btn_no = QPushButton("No")
            btn_no.clicked.connect(self.reject_no)
            btn_layout.addWidget(btn_yes)
            btn_layout.addWidget(btn_no)
        else:
            btn_ok = QPushButton("OK")
            btn_ok.clicked.connect(self.accept_ok)
            btn_layout.addWidget(btn_ok)

        layout.addLayout(btn_layout)

    def accept_yes(self) -> None:
        self.result = QMessageBox.StandardButton.Yes
        self.accept()

    def reject_no(self) -> None:
        self.result = QMessageBox.StandardButton.No
        self.reject()

    def accept_ok(self) -> None:
        self.result = QMessageBox.StandardButton.Ok
        self.accept()

    def exec(self) -> QMessageBox.StandardButton:
        super().exec()
        return self.result


# =========================================================================
# MAIN CLASS
# =========================================================================
class TabMission(QWidget):
    """UI Tab for managing UAV mission configurations.

    Provides fields for trajectory planning, waypoint management, environmental
    wind modeling, and a 3D matplotlib visualization canvas.

    Attributes:
        main_window (QWidget): Reference to the parent main window.
    """

    def __init__(self, main_window_ref: QWidget) -> None:
        """Initializes the mission configuration tab.

        Args:
            main_window_ref (QWidget): Reference to the main application window.
        """
        super().__init__()
        self.main_window = main_window_ref
        self._last_selected_row: int = -1
        self.ax: Any = None  # Dynamically holds matplotlib 2D or 3D axes

        self._setup_dark_theme()
        self._build_ui()
        self._setup_shortcuts()

    def _setup_dark_theme(self) -> None:
        """Applies dark theme styling specifically for the mission tab."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setPalette(palette)

        self.setStyleSheet(
            "QGroupBox { border: 1px solid #555555; border-radius: 4px; margin-top: 1ex; "
            "padding-top: 10px; }\n"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; "
            "padding: 0 3px; color: #dddddd; }"
        )

    def _setup_shortcuts(self) -> None:
        """Configures keyboard shortcuts for the tab's main actions."""
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_new.activated.connect(self.reset_mission)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_mission)

        self.shortcut_load = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_load.activated.connect(self.load_mission)

        self.shortcut_help = QShortcut(QKeySequence("F1"), self)
        self.shortcut_help.activated.connect(self.show_help_window)

        self.shortcut_load_wp = QShortcut(QKeySequence("Ctrl+L"), self)
        self.shortcut_load_wp.activated.connect(self.load_waypoints_from_file)

        self.shortcut_add_wp = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_add_wp.activated.connect(self.add_manual_waypoint)

        self.shortcut_edit_wp = QShortcut(QKeySequence("F2"), self)
        self.shortcut_edit_wp.activated.connect(self.edit_waypoint)

        self.shortcut_dup_wp = QShortcut(QKeySequence("Ctrl+D"), self)
        self.shortcut_dup_wp.activated.connect(self.duplicate_waypoint)

        self.shortcut_up_wp = QShortcut(QKeySequence("Alt+Up"), self)
        self.shortcut_up_wp.activated.connect(self.move_waypoint_up)

        self.shortcut_down_wp = QShortcut(QKeySequence("Alt+Down"), self)
        self.shortcut_down_wp.activated.connect(self.move_waypoint_down)

        self.shortcut_del_wp = QShortcut(QKeySequence("Del"), self)
        self.shortcut_del_wp.activated.connect(self.remove_waypoint)

    def show_help_window(self) -> None:
        """Instantiates and displays the mission help dialog."""
        self.help_dialog = MissionHelpDialog(self)
        self.help_dialog.show()

    def _build_ui(self) -> None:
        """Constructs the master layout including the toolbar, panels, and graph."""
        main_tab_layout = QVBoxLayout(self)
        main_tab_layout.setSpacing(10)

        main_tab_layout.addLayout(self._build_toolbar())

        split_layout = QHBoxLayout()
        split_layout.setSpacing(15)

        # --- LEFT SIDE: MASTER CONTAINER ---
        self.left_master_container_widget = QWidget()
        left_master_container = QVBoxLayout(self.left_master_container_widget)
        left_master_container.setContentsMargins(0, 0, 0, 0)

        left_master_container.addWidget(self._build_waypoints_group())
        left_master_container.addWidget(self._build_trajectory_group())
        left_master_container.addWidget(self._build_environment_group())
        left_master_container.addStretch()

        split_layout.addWidget(self.left_master_container_widget)
        split_layout.addWidget(self._build_visualizer_group(), 1)
        main_tab_layout.addLayout(split_layout)

        self.combo_wind_type.currentIndexChanged.connect(self.toggle_gust_input)
        self.toggle_gust_input()
        self.toggle_toolbar()
        self.update_plot()

    def _build_toolbar(self) -> QHBoxLayout:
        """Constructs the top toolbar with global actions.

        Returns:
            QHBoxLayout: The layout containing the toolbar buttons.
        """
        global_toolbar_layout = QHBoxLayout()

        self.btn_new_mission = QPushButton()
        self.btn_new_mission.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        )
        self.btn_new_mission.setToolTip("New Mission (Ctrl+R)")
        self.btn_new_mission.clicked.connect(self.reset_mission)

        self.btn_save_mission = QPushButton()
        self.btn_save_mission.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon)
        )
        self.btn_save_mission.setToolTip("Save Current Mission (Ctrl+S)")
        self.btn_save_mission.clicked.connect(self.save_mission)

        self.btn_load_mission = QPushButton()
        self.btn_load_mission.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )
        self.btn_load_mission.setToolTip("Load Saved Mission (Ctrl+O)")
        self.btn_load_mission.clicked.connect(self.load_mission)

        self.btn_help = QPushButton()
        self.btn_help.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        )
        self.btn_help.setToolTip("Mission Parameters Guide (F1)")
        self.btn_help.clicked.connect(self.show_help_window)

        global_toolbar_layout.addWidget(self.btn_new_mission)
        global_toolbar_layout.addWidget(self.btn_save_mission)
        global_toolbar_layout.addWidget(self.btn_load_mission)
        global_toolbar_layout.addWidget(self.btn_help)
        global_toolbar_layout.addStretch()

        self.lbl_sim_status = QLabel("Status: Waiting for Configuration")
        self.lbl_sim_status.setStyleSheet("font-size: 13px; color: #aaaaaa;")
        self.lbl_sim_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.btn_run_sim = QPushButton(" Run Simulation")
        self.btn_run_sim.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_run_sim.setToolTip("Execute Dynamic Simulation")
        self.btn_run_sim.clicked.connect(self.start_simulation)

        global_toolbar_layout.addWidget(self.lbl_sim_status)
        global_toolbar_layout.addWidget(self.btn_run_sim)
        return global_toolbar_layout

    def _build_waypoints_group(self) -> QGroupBox:
        """Constructs the waypoints table and its control inputs.

        Returns:
            QGroupBox: The initialized group box for waypoint configuration.
        """
        waypoints_group = QGroupBox("1. Waypoints Configuration")
        waypoints_group.setFixedSize(850, 400)
        wp_internal_layout = QHBoxLayout()

        wp_inputs_container = QWidget()
        wp_inputs_container.setFixedWidth(300)
        wp_inputs_layout = QVBoxLayout(wp_inputs_container)
        wp_inputs_layout.setContentsMargins(0, 0, 10, 0)

        coord_layout = QVBoxLayout()
        self.radio_cartesian = QRadioButton("Cartesian (X, Y, Z)")
        self.radio_geodesic = QRadioButton("Geodesic (Lat, Lon, Alt)")
        self.radio_cartesian.setChecked(True)
        
        self.radio_cartesian.toggled.connect(self.update_coordinate_labels)
        self.radio_cartesian.toggled.connect(self.update_plot)
        self.radio_geodesic.toggled.connect(self.update_plot)
        
        coord_layout.addWidget(self.radio_cartesian)
        coord_layout.addWidget(self.radio_geodesic)
        wp_inputs_layout.addLayout(coord_layout)

        btn_load_file = QPushButton(" Load Coordinates")
        btn_load_file.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        btn_load_file.clicked.connect(self.load_waypoints_from_file)
        wp_inputs_layout.addWidget(btn_load_file)

        form_layout = QFormLayout()
        self.label_x = QLabel("X [m]:")
        self.label_x.setFixedWidth(90)
        
        self.label_y = QLabel("Y [m]:")
        self.label_y.setFixedWidth(90)
        
        self.label_z = QLabel("Z [m]:")
        self.label_z.setFixedWidth(90)

        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-9999999.99, 9999999.99)
        self.spin_x.setDecimals(6)
        
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-9999999.99, 9999999.99)
        self.spin_y.setDecimals(6)
        
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(-9999.99, 9999.99)
        self.spin_z.setDecimals(2)

        form_layout.addRow(self.label_x, self.spin_x)
        form_layout.addRow(self.label_y, self.spin_y)
        form_layout.addRow(self.label_z, self.spin_z)
        wp_inputs_layout.addLayout(form_layout)

        btn_add = QPushButton(" Add Point")
        btn_add.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        btn_add.clicked.connect(self.add_manual_waypoint)
        wp_inputs_layout.addWidget(btn_add)
        wp_inputs_layout.addStretch()

        wp_table_layout = QVBoxLayout()
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 5)

        self.btn_edit = QPushButton()
        self.btn_edit.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self.btn_dup = QPushButton()
        self.btn_dup.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        self.btn_up = QPushButton()
        self.btn_up.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.btn_down = QPushButton()
        self.btn_down.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.btn_rem = QPushButton()
        self.btn_rem.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )

        self.btn_edit.clicked.connect(self.edit_waypoint)
        self.btn_dup.clicked.connect(self.duplicate_waypoint)
        self.btn_up.clicked.connect(self.move_waypoint_up)
        self.btn_down.clicked.connect(self.move_waypoint_down)
        self.btn_rem.clicked.connect(self.remove_waypoint)

        toolbar_layout.addWidget(self.btn_edit)
        toolbar_layout.addWidget(self.btn_dup)
        toolbar_layout.addWidget(self.btn_up)
        toolbar_layout.addWidget(self.btn_down)
        toolbar_layout.addWidget(self.btn_rem)
        wp_table_layout.addWidget(self.toolbar_widget)

        self.table_waypoints = QTableWidget(0, 3)
        self.table_waypoints.setHorizontalHeaderLabels(["X [m]", "Y [m]", "Z [m]"])
        self.table_waypoints.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table_waypoints.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_waypoints.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table_waypoints.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed
        )
        self.table_waypoints.itemSelectionChanged.connect(self.toggle_toolbar)
        self.table_waypoints.itemChanged.connect(self.update_plot)
        wp_table_layout.addWidget(self.table_waypoints)

        wp_internal_layout.addWidget(wp_inputs_container)
        wp_internal_layout.addLayout(wp_table_layout)
        waypoints_group.setLayout(wp_internal_layout)
        return waypoints_group

    def _build_trajectory_group(self) -> QGroupBox:
        """Constructs the UI group for trajectory planning and speed limits.

        Returns:
            QGroupBox: The initialized group box for trajectory configuration.
        """
        trajectory_group = QGroupBox("2. Trajectory Planning")
        trajectory_group.setFixedSize(850, 150)
        traj_layout = QHBoxLayout()

        traj_form_left = QFormLayout()
        self.combo_traj_type = QComboBox()
        self.combo_traj_type.addItems(["5th Degree Polynomial"])

        self.spin_speed_xy = QDoubleSpinBox()
        self.spin_speed_xy.setRange(0.1, 30.0)
        self.spin_speed_xy.setValue(10.0)
        self.spin_speed_xy.setSuffix(" m/s")
        
        self.spin_speed_z = QDoubleSpinBox()
        self.spin_speed_z.setRange(0.1, 15.0)
        self.spin_speed_z.setValue(5.0)
        self.spin_speed_z.setSuffix(" m/s")

        traj_form_left.addRow("Generation Type:", self.combo_traj_type)
        traj_form_left.addRow("Max Speed (XY):", self.spin_speed_xy)
        traj_form_left.addRow("Max Speed (Z):", self.spin_speed_z)

        traj_form_right = QFormLayout()
        self.combo_yaw_mode = QComboBox()
        self.combo_yaw_mode.addItems(
            ["Free (Initial Fixed)", "Forward (Tangent)", "Target (Fixed)"]
        )
        self.combo_yaw_mode.currentIndexChanged.connect(self.toggle_yaw_target)

        self.yaw_target_container = QWidget()
        yaw_target_layout = QHBoxLayout(self.yaw_target_container)
        yaw_target_layout.setContentsMargins(0, 0, 0, 0)

        self.spin_yaw_x = QDoubleSpinBox()
        self.spin_yaw_x.setDecimals(1)
        self.spin_yaw_x.setPrefix("X: ")
        
        self.spin_yaw_y = QDoubleSpinBox()
        self.spin_yaw_y.setDecimals(1)
        self.spin_yaw_y.setPrefix("Y: ")
        
        self.spin_yaw_z = QDoubleSpinBox()
        self.spin_yaw_z.setDecimals(1)
        self.spin_yaw_z.setPrefix("Z: ")
        
        yaw_target_layout.addWidget(self.spin_yaw_x)
        yaw_target_layout.addWidget(self.spin_yaw_y)
        yaw_target_layout.addWidget(self.spin_yaw_z)

        traj_form_right.addRow("Yaw Mode:", self.combo_yaw_mode)
        traj_form_right.addRow("Target Coord.:", self.yaw_target_container)

        traj_layout.addLayout(traj_form_left)
        traj_layout.addLayout(traj_form_right)
        trajectory_group.setLayout(traj_layout)
        self.toggle_yaw_target()
        
        return trajectory_group

    def _build_environment_group(self) -> QGroupBox:
        """Constructs the UI group for wind modeling and live weather API.

        Returns:
            QGroupBox: The initialized group box for environment configuration.
        """
        env_group = QGroupBox("3. Environment & Location Parameters")
        env_group.setFixedSize(850, 250)
        env_layout = QVBoxLayout()

        wind_layout = QHBoxLayout()
        wind_form_left = QFormLayout()
        wind_form_right = QFormLayout()

        self.combo_wind_type = QComboBox()
        self.combo_wind_type.addItems(
            ["None", "Constant", "Dryden Gust", "Dryden Low-Pass", "Sinusoid"]
        )
        self.spin_wind_mag = QDoubleSpinBox()
        self.spin_wind_mag.setRange(0, 100)
        self.spin_wind_mag.setSuffix(" m/s")

        lbl_heading = QLabel("Heading [&deg;]:")
        lbl_heading.setTextFormat(Qt.TextFormat.RichText)
        lbl_elev = QLabel("Elevation [&deg;]:")
        lbl_elev.setTextFormat(Qt.TextFormat.RichText)

        self.spin_wind_head = QDoubleSpinBox()
        self.spin_wind_head.setRange(-360, 360)
        self.spin_wind_head.setSuffix(" °")
        
        self.spin_wind_elev = QDoubleSpinBox()
        self.spin_wind_elev.setRange(-90, 90)
        self.spin_wind_elev.setSuffix(" °")
        
        self.spin_wind_gust = QDoubleSpinBox()
        self.spin_wind_gust.setRange(0, 100)
        self.spin_wind_gust.setSuffix(" m/s")

        wind_form_left.addRow("Wind Type:", self.combo_wind_type)
        wind_form_left.addRow("Base Magnitude:", self.spin_wind_mag)
        wind_form_left.addRow(lbl_heading, self.spin_wind_head)
        
        wind_form_right.addRow(lbl_elev, self.spin_wind_elev)
        wind_form_right.addRow("Max Gust:", self.spin_wind_gust)
        
        wind_layout.addLayout(wind_form_left)
        wind_layout.addLayout(wind_form_right)
        env_layout.addLayout(wind_layout)

        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555555;")
        env_layout.addWidget(line)

        loc_layout = QHBoxLayout()
        loc_form = QFormLayout()
        
        self.spin_loc_lat = QDoubleSpinBox()
        self.spin_loc_lat.setRange(-90, 90)
        self.spin_loc_lat.setDecimals(6)
        self.spin_loc_lat.setValue(-22.739000)  # Default for Piracicaba

        self.spin_loc_lon = QDoubleSpinBox()
        self.spin_loc_lon.setRange(-180, 180)
        self.spin_loc_lon.setDecimals(6)
        self.spin_loc_lon.setValue(-47.646000)  # Default for Piracicaba

        loc_form.addRow("Latitude:", self.spin_loc_lat)
        loc_form.addRow("Longitude:", self.spin_loc_lon)

        btn_fetch_weather = QPushButton(" OpenWeather Data")
        btn_fetch_weather.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )
        btn_fetch_weather.clicked.connect(self.fetch_weather_data)

        loc_layout.addLayout(loc_form)
        loc_layout.addWidget(btn_fetch_weather)
        loc_layout.addStretch()
        
        env_layout.addLayout(loc_layout)
        env_group.setLayout(env_layout)
        return env_group

    def _build_visualizer_group(self) -> QGroupBox:
        """Constructs the Matplotlib 3D trajectory visualization area.

        Returns:
            QGroupBox: The initialized group box holding the canvas.
        """
        visualizer_group = QGroupBox("Trajectory Visualizer")
        vis_layout = QVBoxLayout()

        vis_controls = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(
            ["3D Perspective", "Top View (XY)", "Side View (YZ)", "Front View (XZ)"]
        )
        self.view_combo.currentIndexChanged.connect(self.update_plot)

        btn_reset_view = QPushButton()
        btn_reset_view.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        btn_reset_view.clicked.connect(self.reset_graph_view)

        vis_controls.addWidget(QLabel("Camera:"))
        vis_controls.addWidget(self.view_combo)
        vis_controls.addStretch()
        vis_controls.addWidget(btn_reset_view)
        vis_layout.addLayout(vis_controls)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.patch.set_facecolor("#2b2b2b")
        vis_layout.addWidget(self.canvas)
        
        visualizer_group.setLayout(vis_layout)
        return visualizer_group

    # =========================================================================
    # PLOTTING LOGIC
    # =========================================================================
    def update_plot(self) -> None:
        """Refreshes the Matplotlib canvas with the current waypoint data."""
        self.figure.clear()
        view_mode = self.view_combo.currentText()
        is_cartesian = self.radio_cartesian.isChecked()
        x_vals, y_vals, z_vals = [], [], []

        for row in range(self.table_waypoints.rowCount()):
            try:
                x_i = self.table_waypoints.item(row, 0)
                y_i = self.table_waypoints.item(row, 1)
                z_i = self.table_waypoints.item(row, 2)
                if x_i and y_i and z_i:
                    x_vals.append(float(x_i.text()))
                    y_vals.append(float(y_i.text()))
                    z_vals.append(float(z_i.text()))
            except ValueError:
                pass

        lbl_x = "X (m)" if is_cartesian else "Latitude (deg)"
        lbl_y = "Y (m)" if is_cartesian else "Longitude (deg)"
        lbl_z = "Z (m)" if is_cartesian else "Altitude (m)"

        if "3D" in view_mode:
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_axis_on()
            self.ax.grid(True, linestyle=":", color="#444444", alpha=0.5)
            self.ax.xaxis.set_pane_color((0, 0, 0, 0))
            self.ax.yaxis.set_pane_color((0, 0, 0, 0))
            self.ax.zaxis.set_pane_color((0, 0, 0, 0))
            self.ax.xaxis.line.set_color("#555555")
            self.ax.yaxis.line.set_color("#555555")
            self.ax.zaxis.line.set_color("#555555")

            self.ax.set_xlabel(lbl_x, labelpad=15, fontsize=12)
            self.ax.set_ylabel(lbl_y, labelpad=15, fontsize=12)
            self.ax.set_zlabel(lbl_z, labelpad=15, fontsize=12)
            self.figure.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
            
            try:
                self.ax.set_box_aspect(None)
            except Exception:
                pass

            if x_vals:
                self.ax.plot(
                    x_vals, y_vals, z_vals, linestyle="-", color="#00d4ff", linewidth=1.5
                )
                self.ax.scatter(x_vals, y_vals, z_vals, color="white", s=25)
                self.ax.plot(
                    [x_vals[0]], [y_vals[0]], [z_vals[0]], 
                    marker="s", color="#4caf50", markersize=8
                )
                if len(x_vals) > 1:
                    self.ax.plot(
                        [x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 
                        marker="X", color="#f44336", markersize=8
                    )
                self.ax.invert_zaxis()
        else:
            self.ax = self.figure.add_subplot(111)
            self.ax.grid(True, linestyle=":", color="#444444", alpha=0.5)
            
            for spine in self.ax.spines.values():
                spine.set_color("#555555")
                
            self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)

            if x_vals:
                if "XY" in view_mode:
                    h, v = x_vals, y_vals
                    self.ax.set_xlabel(lbl_x, fontsize=12)
                    self.ax.set_ylabel(lbl_y, fontsize=12)
                elif "YZ" in view_mode:
                    h, v = y_vals, z_vals
                    self.ax.set_xlabel(lbl_y, fontsize=12)
                    self.ax.set_ylabel(lbl_z, fontsize=12)
                else:
                    h, v = x_vals, z_vals
                    self.ax.set_xlabel(lbl_x, fontsize=12)
                    self.ax.set_ylabel(lbl_z, fontsize=12)

                self.ax.plot(
                    h, v, linestyle="-", color="#00d4ff", marker="o", 
                    markerfacecolor="white"
                )
                self.ax.plot([h[0]], [v[0]], marker="s", color="#4caf50", markersize=8)
                
                if len(h) > 1:
                    self.ax.plot(
                        [h[-1]], [v[-1]], marker="X", color="#f44336", markersize=8
                    )
                    
                if "YZ" in view_mode or "XZ" in view_mode:
                    self.ax.invert_yaxis()

        self.ax.set_facecolor("#2b2b2b")
        self.canvas.draw()

    # =========================================================================
    # MATHEMATICAL ENGINE INTEGRATION AND GENERAL LOGIC
    # =========================================================================
    def start_simulation(self) -> None:
        """Validates configuration and triggers the ROS 2 simulation."""
        mission_data = self._collect_mission_data()
        aircraft_data = self._collect_aircraft_data()

        if not mission_data.get("waypoints"):
            msg = CustomMessageBox(
                "Warning",
                "You need to define at least one Waypoint in the Mission tab.",
                msg_type="info",
                parent=self,
            )
            msg.exec()
            return

        if aircraft_data.get("mass", 0.0) <= 0.01:
            msg = CustomMessageBox(
                "Error",
                "Aircraft parameters are empty!",
                "Please go to the 'Aircraft Parameters' tab and load a valid model.",
                msg_type="error",
                parent=self,
            )
            msg.exec()
            return

        try:
            self.export_to_ros_yaml()
            if hasattr(self.main_window, "tab_aircraft"):
                self.main_window.tab_aircraft.export_to_ros_yaml()
        except Exception as err:
            msg = CustomMessageBox(
                "Export Error",
                "Could not export parameters to ROS YAML.",
                str(err),
                msg_type="error",
                parent=self,
            )
            msg.exec()
            return

        self.lbl_sim_status.setText("Status: ROS Mission Triggered!")
        self.lbl_sim_status.setStyleSheet(
            "color: #00ff00; font-size: 13px; font-weight: bold;"
        )

    def _collect_aircraft_data(self) -> Dict[str, float]:
        """Retrieves aircraft parameters from the sibling aircraft tab.

        Returns:
            Dict[str, float]: The dictionary of aircraft parameters.
        """
        data = {}
        tab_ac = getattr(self.main_window, "tab_aircraft", None)
        if tab_ac:
            for key, spin in tab_ac.aircraft_spins.items():
                data[key] = spin.value()
        return data

    def _collect_mission_data(self) -> Dict[str, Any]:
        """Aggregates all mission inputs from the UI into a dictionary.

        Returns:
            Dict[str, Any]: A serialized dictionary of the mission profile.
        """
        data = {
            "coordinate_system": "cartesian"
            if self.radio_cartesian.isChecked()
            else "geodesic",
            "wind": {
                "type": self.combo_wind_type.currentText(),
                "magnitude": self.spin_wind_mag.value(),
                "heading": self.spin_wind_head.value(),
                "elevation": self.spin_wind_elev.value(),
                "gust": self.spin_wind_gust.value(),
            },
            "location": {
                "latitude": self.spin_loc_lat.value(),
                "longitude": self.spin_loc_lon.value(),
            },
            "trajectory": {
                "type": self.combo_traj_type.currentText(),
                "speed_xy": self.spin_speed_xy.value(),
                "speed_z": self.spin_speed_z.value(),
                "yaw_mode": self.combo_yaw_mode.currentText(),
                "yaw_target": {
                    "x": self.spin_yaw_x.value(),
                    "y": self.spin_yaw_y.value(),
                    "z": self.spin_yaw_z.value(),
                },
            },
            "waypoints": [],
        }
        for row in range(self.table_waypoints.rowCount()):
            try:
                x = float(self.table_waypoints.item(row, 0).text())
                y = float(self.table_waypoints.item(row, 1).text())
                z = float(self.table_waypoints.item(row, 2).text())
                data["waypoints"].append({"x": x, "y": y, "z": z})
            except (ValueError, AttributeError):
                pass
        return data

    def reset_mission(self) -> None:
        """Prompts the user and resets all mission parameters to default."""
        msg = CustomMessageBox(
            "New Mission",
            "Do you want to reset everything?",
            "Unsaved data will be lost.",
            msg_type="question",
            parent=self,
        )
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.table_waypoints.blockSignals(True)
            self.table_waypoints.setRowCount(0)
            
            self.radio_cartesian.setChecked(True)
            self.combo_wind_type.setCurrentIndex(0)
            
            self.spin_wind_mag.setValue(0.0)
            self.spin_wind_head.setValue(0.0)
            self.spin_wind_elev.setValue(0.0)
            self.spin_wind_gust.setValue(0.0)
            
            self.spin_loc_lat.setValue(-22.739000)
            self.spin_loc_lon.setValue(-47.646000)
            
            self.table_waypoints.blockSignals(False)

            self.lbl_sim_status.setText("Status: Waiting for Configuration")
            self.lbl_sim_status.setStyleSheet("font-size: 13px; color: #aaaaaa;")
            self.update_plot()

    def save_mission(self) -> None:
        """Saves the current mission as a JSON file and exports to ROS 2."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mission", str(MISSIONS_DIR), "Mission Files (*.json)"
        )
        if not file_path:
            return
            
        if not file_path.endswith(".json"):
            file_path += ".json"
            
        mission_data = self._collect_mission_data()
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(mission_data, file, indent=4)
                
            self.export_to_ros_yaml()
            
            msg = CustomMessageBox(
                "Success",
                "Mission saved successfully!",
                f"File saved at:\n{file_path}",
                msg_type="success",
                parent=self,
            )
            msg.exec()
        except Exception as err:
            msg = CustomMessageBox(
                "Save Error", "Could not save the file.", str(err), 
                msg_type="error", parent=self
            )
            msg.exec()

    def export_to_ros_yaml(self) -> None:
        """Exports the current mission parameters into ROS 2 YAML configs."""
        waypoints_list = []
        for row in range(self.table_waypoints.rowCount()):
            try:
                x = float(self.table_waypoints.item(row, 0).text())
                y = float(self.table_waypoints.item(row, 1).text())
                z = float(self.table_waypoints.item(row, 2).text())
                waypoints_list.extend([x, y, z])
            except (ValueError, AttributeError):
                pass

        yaw_mode_ui = self.combo_yaw_mode.currentText()
        if "Forward" in yaw_mode_ui:
            yaw_mode = "forward"
        elif "Target" in yaw_mode_ui:
            yaw_mode = "target"
        else:
            yaw_mode = "free"

        ros_data = {
            "trajectory_node": {
                "ros__parameters": {
                    "waypoints": waypoints_list,
                    "generation_type": self.combo_traj_type.currentText(),
                    "max_speed_xy": self.spin_speed_xy.value(),
                    "max_speed_z": self.spin_speed_z.value(),
                    "yaw_mode": yaw_mode,
                    "yaw_target": [
                        self.spin_yaw_x.value(),
                        self.spin_yaw_y.value(),
                        self.spin_yaw_z.value(),
                    ],
                    "wind_type": self.combo_wind_type.currentText(),
                    "wind_magnitude": self.spin_wind_mag.value(),
                    "wind_heading": self.spin_wind_head.value(),
                    "wind_elevation": self.spin_wind_elev.value(),
                    "wind_gust": self.spin_wind_gust.value(),
                    "latitude": self.spin_loc_lat.value(),
                    "longitude": self.spin_loc_lon.value(),
                }
            }
        }

        yaml_path_src = ROS_CONFIG_DIR / "mission_params.yaml"
        with open(yaml_path_src, "w", encoding="utf-8") as file:
            yaml.dump(ros_data, file, default_flow_style=None, sort_keys=False)

        if ROS_INSTALL_DIR and ROS_INSTALL_DIR.exists():
            yaml_path_install = ROS_INSTALL_DIR / "mission_params.yaml"
            with open(yaml_path_install, "w", encoding="utf-8") as file:
                yaml.dump(ros_data, file, default_flow_style=None, sort_keys=False)

    def load_mission(self) -> None:
        """Prompts the user to select and load a mission JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mission", str(MISSIONS_DIR), "Mission Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.table_waypoints.blockSignals(True)

            if data.get("coordinate_system") == "cartesian":
                self.radio_cartesian.setChecked(True)
            else:
                self.radio_geodesic.setChecked(True)

            wind_data = data.get("wind", {})
            self.combo_wind_type.setCurrentText(wind_data.get("type", "None"))
            self.spin_wind_mag.setValue(wind_data.get("magnitude", 0.0))
            self.spin_wind_head.setValue(wind_data.get("heading", 0.0))
            self.spin_wind_elev.setValue(wind_data.get("elevation", 0.0))
            self.spin_wind_gust.setValue(wind_data.get("gust", 0.0))

            loc_data = data.get("location", {})
            self.spin_loc_lat.setValue(loc_data.get("latitude", 0.0))
            self.spin_loc_lon.setValue(loc_data.get("longitude", 0.0))

            self.table_waypoints.setRowCount(0)
            for wp in data.get("waypoints", []):
                self._add_row_to_table(
                    wp.get("x", 0.0), wp.get("y", 0.0), wp.get("z", 0.0)
                )

            traj_data = data.get("trajectory", {})
            self.combo_traj_type.setCurrentText(
                traj_data.get("type", "5th Degree Polynomial")
            )
            self.spin_speed_xy.setValue(traj_data.get("speed_xy", 10.0))
            self.spin_speed_z.setValue(traj_data.get("speed_z", 5.0))
            self.combo_yaw_mode.setCurrentText(
                traj_data.get("yaw_mode", "Free (Initial Fixed)")
            )

            yaw_t = traj_data.get("yaw_target", {})
            self.spin_yaw_x.setValue(yaw_t.get("x", 0.0))
            self.spin_yaw_y.setValue(yaw_t.get("y", 0.0))
            self.spin_yaw_z.setValue(yaw_t.get("z", 0.0))

            self.table_waypoints.blockSignals(False)
            
            self.lbl_sim_status.setText("Status: File Loaded")
            self.lbl_sim_status.setStyleSheet("color: #aaaaaa; font-size: 13px;")
            self.update_plot()

            filename = os.path.basename(file_path)
            msg = CustomMessageBox(
                "Success",
                "Mission loaded!",
                f"Model '{filename}' loaded successfully.",
                msg_type="success",
                parent=self,
            )
            msg.exec()

        except Exception as err:
            self.table_waypoints.blockSignals(False)
            msg = CustomMessageBox(
                "Read Error",
                "The selected file is not a valid Mission.",
                str(err),
                msg_type="error",
                parent=self,
            )
            msg.exec()

    def toggle_gust_input(self) -> None:
        """Enables or disables gust parameter fields based on wind type."""
        wind_type = self.combo_wind_type.currentText()
        if "None" in wind_type or "Constant" in wind_type:
            self.spin_wind_gust.setEnabled(False)
            self.spin_wind_gust.setValue(0.0)
        else:
            self.spin_wind_gust.setEnabled(True)

    # -------------------------------------------------------------------------
    # OPENWEATHER API INTEGRATION
    # -------------------------------------------------------------------------
    def fetch_weather_data(self) -> None:
        """Fetches live wind and weather data based on current coordinates."""
        lat = self.spin_loc_lat.value()
        lon = self.spin_loc_lon.value()

        if lat == 0.0 and lon == 0.0:
            msg = CustomMessageBox(
                "Warning",
                "Please enter valid Latitude and Longitude.",
                msg_type="info",
                parent=self,
            )
            msg.exec()
            return

        self.setCursor(Qt.CursorShape.WaitCursor)

        api_key = "1460b3ee5d76829bad64d11cdefac23a"
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )

        try:
            response = requests.get(url, timeout=5)
            self.unsetCursor()

            if response.status_code == 200:
                data = response.json()

                wind_speed = data["wind"].get("speed", 0.0)
                wind_deg = data["wind"].get("deg", 0.0)
                wind_gust = data["wind"].get("gust", 0.0)
                sky = data["weather"][0].get("description", "N/A")
                temp = data["main"].get("temp", "N/A")

                self.combo_wind_type.setCurrentText("Constant")
                if wind_gust > 0.0:
                    self.combo_wind_type.setCurrentText("Dryden Gust")

                self.spin_wind_mag.setValue(float(wind_speed))
                self.spin_wind_head.setValue(float(wind_deg))
                self.spin_wind_gust.setValue(float(wind_gust))

                detail = (
                    f"Weather: {sky.title()}\n"
                    f"Temperature: {temp}°C\n"
                    f"Wind Speed: {wind_speed} m/s\n"
                    f"Wind Gust: {wind_gust} m/s"
                )
                msg = CustomMessageBox(
                    "OpenWeather Success",
                    "Live weather data fetched and applied!",
                    detail,
                    msg_type="success",
                    parent=self,
                )
                msg.exec()
            else:
                msg = CustomMessageBox(
                    "API Error",
                    f"Server returned status code: {response.status_code}",
                    msg_type="error",
                    parent=self,
                )
                msg.exec()
        except Exception as err:
            self.unsetCursor()
            msg = CustomMessageBox(
                "Connection Error",
                "Failed to connect to OpenWeather API.",
                str(err),
                msg_type="error",
                parent=self,
            )
            msg.exec()

    def update_coordinate_labels(self) -> None:
        """Updates table headers depending on the coordinate system chosen."""
        if self.radio_cartesian.isChecked():
            self.label_x.setText("X [m]:")
            self.label_y.setText("Y [m]:")
            self.label_z.setText("Z [m]:")
            self.table_waypoints.setHorizontalHeaderLabels(["X [m]", "Y [m]", "Z [m]"])
        else:
            self.label_x.setText("Latitude:")
            self.label_y.setText("Longitude:")
            self.label_z.setText("Altitude [m]:")
            self.table_waypoints.setHorizontalHeaderLabels(
                ["Latitude", "Longitude", "Altitude [m]"]
            )

    def toggle_toolbar(self) -> None:
        """Enables or disables table toolbar buttons based on selection count."""
        selected_rows = set(item.row() for item in self.table_waypoints.selectedItems())
        count = len(selected_rows)
        
        self.btn_edit.setEnabled(count == 1)
        self.btn_up.setEnabled(count == 1)
        self.btn_down.setEnabled(count == 1)
        self.btn_dup.setEnabled(count >= 1)
        self.btn_rem.setEnabled(count >= 1)

    def handle_item_click(self, item: QTableWidgetItem) -> None:
        """Manages custom row selection logic.

        Args:
            item (QTableWidgetItem): The item clicked in the table.
        """
        row = item.row()
        if self._last_selected_row == row:
            self.table_waypoints.clearSelection()
            self._last_selected_row = -1
        else:
            self._last_selected_row = row

    def _add_row_to_table(self, x: float, y: float, z: float) -> None:
        """Inserts a new coordinate row into the waypoint table.

        Args:
            x (float): X coordinate or Latitude.
            y (float): Y coordinate or Longitude.
            z (float): Z coordinate or Altitude.
        """
        row = self.table_waypoints.rowCount()
        self.table_waypoints.insertRow(row)
        dec = 2 if self.radio_cartesian.isChecked() else 6
        
        self.table_waypoints.setItem(row, 0, QTableWidgetItem(f"{x:.{dec}f}"))
        self.table_waypoints.setItem(row, 1, QTableWidgetItem(f"{y:.{dec}f}"))
        self.table_waypoints.setItem(row, 2, QTableWidgetItem(f"{z:.2f}"))

    def add_manual_waypoint(self) -> None:
        """Adds a waypoint from the spinbox inputs to the table."""
        self._add_row_to_table(
            self.spin_x.value(), self.spin_y.value(), self.spin_z.value()
        )
        self.spin_x.setValue(0.0)
        self.spin_y.setValue(0.0)
        self.spin_z.setValue(0.0)

    def edit_waypoint(self) -> None:
        """Triggers the edit mode for the currently selected waypoint."""
        current_row = self.table_waypoints.currentRow()
        if current_row >= 0:
            self.table_waypoints.editItem(self.table_waypoints.item(current_row, 0))

    def remove_waypoint(self) -> None:
        """Removes the selected waypoints from the table."""
        selected_items = self.table_waypoints.selectedItems()
        if not selected_items:
            return
            
        self.table_waypoints.blockSignals(True)
        rows_to_remove = sorted(
            list(set(item.row() for item in selected_items)), reverse=True
        )
        
        for row in rows_to_remove:
            self.table_waypoints.removeRow(row)
            
        self.table_waypoints.clearSelection()
        self.table_waypoints.blockSignals(False)
        self.update_plot()

    def duplicate_waypoint(self) -> None:
        """Duplicates the selected waypoints and inserts them below."""
        selected_items = self.table_waypoints.selectedItems()
        if not selected_items:
            return
            
        self.table_waypoints.blockSignals(True)
        rows_to_duplicate = sorted(list(set(item.row() for item in selected_items)))
        insert_position = rows_to_duplicate[-1] + 1
        dec = 2 if self.radio_cartesian.isChecked() else 6
        
        for row in reversed(rows_to_duplicate):
            x = float(self.table_waypoints.item(row, 0).text())
            y = float(self.table_waypoints.item(row, 1).text())
            z = float(self.table_waypoints.item(row, 2).text())
            
            self.table_waypoints.insertRow(insert_position)
            self.table_waypoints.setItem(
                insert_position, 0, QTableWidgetItem(f"{x:.{dec}f}")
            )
            self.table_waypoints.setItem(
                insert_position, 1, QTableWidgetItem(f"{y:.{dec}f}")
            )
            self.table_waypoints.setItem(
                insert_position, 2, QTableWidgetItem(f"{z:.2f}")
            )
            
        self.table_waypoints.blockSignals(False)
        self.update_plot()

    def move_waypoint_up(self) -> None:
        """Moves the selected waypoint up one row in the table."""
        current_row = self.table_waypoints.currentRow()
        if current_row > 0:
            self._swap_rows(current_row, current_row - 1)
            self.table_waypoints.setCurrentCell(current_row - 1, 0)

    def move_waypoint_down(self) -> None:
        """Moves the selected waypoint down one row in the table."""
        current_row = self.table_waypoints.currentRow()
        if current_row >= 0 and current_row < self.table_waypoints.rowCount() - 1:
            self._swap_rows(current_row, current_row + 1)
            self.table_waypoints.setCurrentCell(current_row + 1, 0)

    def _swap_rows(self, row1: int, row2: int) -> None:
        """Swaps the data of two rows within the waypoint table.

        Args:
            row1 (int): The index of the first row.
            row2 (int): The index of the second row.
        """
        for col in range(self.table_waypoints.columnCount()):
            item1 = self.table_waypoints.takeItem(row1, col)
            item2 = self.table_waypoints.takeItem(row2, col)
            self.table_waypoints.setItem(row1, col, item2)
            self.table_waypoints.setItem(row2, col, item1)

    def load_waypoints_from_file(self) -> None:
        """Loads a batch of coordinates from a CSV or JSON file into the table."""
        try:
            from paths import WAYPOINTS_DIR
            base_dir = str(WAYPOINTS_DIR)
        except ImportError:
            base_dir = str(MISSIONS_DIR)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Waypoints file", base_dir, "Files (*.csv *.json)"
        )
        if not file_path:
            return

        self.table_waypoints.blockSignals(True)
        try:
            if file_path.endswith(".csv"):
                with open(file_path, mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        x = row.get("X", row.get("lat", 0.0))
                        y = row.get("Y", row.get("lon", 0.0))
                        z = row.get("Z", row.get("alt", 0.0))
                        self._add_row_to_table(float(x), float(y), float(z))
            elif file_path.endswith(".json"):
                with open(file_path, mode="r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, dict) and "waypoints" in data:
                        data = data["waypoints"]
                    for item in data:
                        x = item.get("X", item.get("lat", item.get("x", 0.0)))
                        y = item.get("Y", item.get("lon", item.get("y", 0.0)))
                        z = item.get("Z", item.get("alt", item.get("z", 0.0)))
                        self._add_row_to_table(float(x), float(y), float(z))
        except Exception as err:
            msg = CustomMessageBox(
                "Read Error", "Could not read the file.", str(err), 
                msg_type="error", parent=self
            )
            msg.exec()
        finally:
            self.table_waypoints.blockSignals(False)
            self.update_plot()

    def reset_graph_view(self) -> None:
        """Resets the 3D perspective to the default angle."""
        if hasattr(self, "ax") and self.view_combo.currentText() == "3D Perspective":
            self.ax.view_init(elev=20, azim=-35)
            self.canvas.draw()

    def toggle_yaw_target(self) -> None:
        """Shows or hides target coordinates based on the selected Yaw Mode."""
        is_target_mode = "Target" in self.combo_yaw_mode.currentText()
        self.yaw_target_container.setVisible(is_target_mode)
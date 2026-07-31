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
from typing import Any, Dict, Optional

import numpy as np
import requests
import yaml
from PySide6.QtCore import Qt
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

# --- PYQTGRAPH OPENGL CONFIGURATION ---
import pyqtgraph as pg
import pyqtgraph.opengl as gl


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
        super().__init__(parent)
        self.setWindowTitle("Documentation - Mission Configuration")
        self.resize(850, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        current_dir = Path(__file__).resolve().parent
        gif_path = current_dir.parent / "assets" / "img" / "tutorial_mission.gif"
        gif_uri = gif_path.as_uri()

        html_content = f"""
        <html>
        <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; 
                   line-height: 1.6; color: #e0e0e0; padding: 15px 25px; }}
            h2 {{ color: #00d4ff; margin-top: 10px; border-bottom: 1px solid #555; 
                 padding-bottom: 5px;}}
            h3 {{ color: #4caf50; margin-top: 25px; margin-bottom: 5px; }}
            ul {{ margin-top: 5px; padding-left: 25px; }}
            li {{ margin-bottom: 8px; }}
            b {{ color: #ffffff; }}
            code {{ background-color: #444; padding: 2px 4px; border-radius: 3px; 
                   color: #ffcc00;}}
            .gif-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #555; border-radius: 8px; }}
        </style>
        </head>
        <body>
        <h2>Mission Configuration Guide</h2>
        <p>This section allows you to define the UAV's flight path, trajectory 
        parameters, and environmental conditions for the integration algorithms.</p>
        
        <div class="gif-container">
            <img src="{gif_uri}" alt="Tutorial de Missão">
        </div>

        <h3>1. Waypoints Configuration</h3>
        <ul>
            <li><b>Coordinate System:</b> Choose between Cartesian (X, Y, Z in meters) 
            or Geodesic (Latitude, Longitude, Altitude).</li>
            <li><b>Adding Points:</b> Manually input coordinates or load a batch from a 
            <code>.csv</code> or <code>.json</code> file.</li>
            <li><b>Table Tools:</b> Select rows to Edit, Duplicate, Reorder, or Remove 
            waypoints dynamically.</li>
        </ul>
        <h3>2. Trajectory Planning</h3>
        <ul>
            <li><b>Generation Type:</b> Mathematical method to interpolate waypoints 
            (e.g., 5th Degree Polynomial for smooth, jerk-free motion profiles).</li>
            <li><b>Max Speed (XY) [m/s]:</b> Maximum horizontal speed allowed.</li>
            <li><b>Max Speed (Z) [m/s]:</b> Maximum vertical speed allowed.</li>
            <li><b>Yaw Mode:</b> Defines the drone's heading profile: <br>
            - <i>Free:</i> keeps the initial yaw.<br>
            - <i>Forward:</i> continuously points the nose to the velocity vector.<br>
            - <i>Target:</i> nose always points to a specific XYZ coordinate lock.</li>
        </ul>
        <h3>3. Environment & Location Parameters</h3>
        <ul>
            <li><b>Data Source Toggle:</b> Choose <i>Manual Configuration</i> to input wind data 
            manually, or <i>Current Real-Time Weather</i> to reveal location settings and sync 
            actual weather conditions. Fields are visibly locked in Real-Time mode to ensure data integrity.</li>
            <li><b>Wind Type:</b> Simulation wind disturbance model.</li>
            <li><b>Base Magnitude [m/s]:</b> Base persistent wind speed.</li>
            <li><b>Heading / Elevation [&deg;]:</b> Wind direction vector angles.</li>
            <li><b>Max Gust [m/s]:</b> Maximum magnitude added dynamically by gusts.</li>
        </ul>
        </body>
        </html>
        """
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
    """UI Tab for managing UAV mission configurations."""

    def __init__(self, main_window_ref: QWidget) -> None:
        super().__init__()
        self.main_window = main_window_ref
        self._last_selected_row: int = -1

        self._setup_dark_theme()
        self._build_ui()
        self._setup_shortcuts()

    def _setup_dark_theme(self) -> None:
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
        self.help_dialog = MissionHelpDialog(self)
        self.help_dialog.show()

    def _build_ui(self) -> None:
        main_tab_layout = QVBoxLayout(self)
        main_tab_layout.setSpacing(10)

        main_tab_layout.addLayout(self._build_toolbar())

        split_layout = QHBoxLayout()
        split_layout.setSpacing(15)

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
        self._toggle_weather_mode()
        self.toggle_toolbar()
        
        # Initial Plot Update
        self.update_plot()

    def _build_toolbar(self) -> QHBoxLayout:
        global_toolbar_layout = QHBoxLayout()

        self.btn_new_mission = QPushButton()
        self.btn_new_mission.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.btn_new_mission.setToolTip("New Mission (Ctrl+R)")
        self.btn_new_mission.clicked.connect(self.reset_mission)

        self.btn_save_mission = QPushButton()
        self.btn_save_mission.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon))
        self.btn_save_mission.setToolTip("Save Current Mission (Ctrl+S)")
        self.btn_save_mission.clicked.connect(self.save_mission)

        self.btn_load_mission = QPushButton()
        self.btn_load_mission.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.btn_load_mission.setToolTip("Load Saved Mission (Ctrl+O)")
        self.btn_load_mission.clicked.connect(self.load_mission)

        self.btn_help = QPushButton()
        self.btn_help.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.btn_help.setToolTip("Mission Parameters Guide (F1)")
        self.btn_help.clicked.connect(self.show_help_window)

        global_toolbar_layout.addWidget(self.btn_new_mission)
        global_toolbar_layout.addWidget(self.btn_save_mission)
        global_toolbar_layout.addWidget(self.btn_load_mission)
        global_toolbar_layout.addWidget(self.btn_help)
        global_toolbar_layout.addStretch()

        self.lbl_sim_status = QLabel("Status: Waiting for Configuration")
        self.lbl_sim_status.setStyleSheet("font-size: 13px; color: #aaaaaa;")
        self.lbl_sim_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.btn_run_sim = QPushButton(" Run Simulation")
        self.btn_run_sim.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_run_sim.setToolTip("Execute Dynamic Simulation")
        self.btn_run_sim.clicked.connect(self.start_simulation)

        global_toolbar_layout.addWidget(self.lbl_sim_status)
        global_toolbar_layout.addWidget(self.btn_run_sim)
        return global_toolbar_layout

    def _build_waypoints_group(self) -> QGroupBox:
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
        
        coord_layout.addWidget(self.radio_cartesian)
        coord_layout.addWidget(self.radio_geodesic)
        wp_inputs_layout.addLayout(coord_layout)

        btn_load_file = QPushButton(" Load Coordinates")
        btn_load_file.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
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
        btn_add.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        btn_add.clicked.connect(self.add_manual_waypoint)
        wp_inputs_layout.addWidget(btn_add)
        wp_inputs_layout.addStretch()

        wp_table_layout = QVBoxLayout()
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 5)

        self.btn_edit = QPushButton()
        self.btn_edit.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.btn_dup = QPushButton()
        self.btn_dup.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))
        self.btn_up = QPushButton()
        self.btn_up.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.btn_down = QPushButton()
        self.btn_down.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        self.btn_rem = QPushButton()
        self.btn_rem.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))

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
        self.table_waypoints.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_waypoints.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_waypoints.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table_waypoints.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table_waypoints.itemSelectionChanged.connect(self.toggle_toolbar)
        self.table_waypoints.itemChanged.connect(self.update_plot)
        wp_table_layout.addWidget(self.table_waypoints)

        wp_internal_layout.addWidget(wp_inputs_container)
        wp_internal_layout.addLayout(wp_table_layout)
        waypoints_group.setLayout(wp_internal_layout)
        return waypoints_group

    def _build_trajectory_group(self) -> QGroupBox:
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
        env_group = QGroupBox("3. Environment & Location Parameters")
        env_group.setFixedWidth(850)
        env_group.setMinimumHeight(200)
        env_layout = QVBoxLayout()

        source_layout = QHBoxLayout()
        self.radio_manual_wind = QRadioButton("Manual Configuration")
        self.radio_api_wind = QRadioButton("Current Real-Time Weather")
        self.radio_manual_wind.setChecked(True)
        
        self.radio_manual_wind.toggled.connect(self._toggle_weather_mode)
        self.radio_api_wind.toggled.connect(self._toggle_weather_mode)
        
        source_layout.addWidget(QLabel("<b>Data Source:</b>"))
        source_layout.addWidget(self.radio_manual_wind)
        source_layout.addWidget(self.radio_api_wind)
        source_layout.addStretch()
        env_layout.addLayout(source_layout)

        self.loc_container = QWidget()
        loc_layout = QHBoxLayout(self.loc_container)
        loc_layout.setContentsMargins(0, 5, 0, 5)

        loc_form = QFormLayout()
        
        self.spin_loc_lat = QDoubleSpinBox()
        self.spin_loc_lat.setRange(-90, 90)
        self.spin_loc_lat.setDecimals(6)
        self.spin_loc_lat.setValue(-22.739000) 

        self.spin_loc_lon = QDoubleSpinBox()
        self.spin_loc_lon.setRange(-180, 180)
        self.spin_loc_lon.setDecimals(6)
        self.spin_loc_lon.setValue(-47.646000) 

        loc_form.addRow("Latitude:", self.spin_loc_lat)
        loc_form.addRow("Longitude:", self.spin_loc_lon)

        self.btn_fetch_weather = QPushButton(" Sync API")
        self.btn_fetch_weather.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_fetch_weather.setToolTip("Fetch real-time weather for these coordinates")
        self.btn_fetch_weather.setFixedHeight(30)
        self.btn_fetch_weather.setStyleSheet(
            "QPushButton { background-color: #3b3b3b; color: #00d4ff; border: 1px solid #00d4ff; border-radius: 4px; padding: 4px 12px; font-weight: bold;}"
            "QPushButton:hover { background-color: #00d4ff; color: #2b2b2b; }"
        )
        self.btn_fetch_weather.clicked.connect(self.fetch_weather_data)

        loc_layout.addLayout(loc_form)
        loc_layout.addWidget(self.btn_fetch_weather)
        loc_layout.addStretch()
        
        env_layout.addWidget(self.loc_container)

        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555555;")
        env_layout.addWidget(line)

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

        env_group.setLayout(env_layout)
        
        return env_group

    def _build_visualizer_group(self) -> QGroupBox:
        """Constructs the OpenGL hardware-accelerated visualization area."""
        visualizer_group = QGroupBox("Trajectory Visualizer (OpenGL)")
        vis_layout = QVBoxLayout()

        vis_controls = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(
            ["3D Perspective", "Top View (XY)", "Side View (YZ)", "Front View (XZ)"]
        )
        self.view_combo.currentIndexChanged.connect(self.set_camera_view)

        btn_reset_view = QPushButton()
        btn_reset_view.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        btn_reset_view.clicked.connect(self.set_camera_view)

        vis_controls.addWidget(QLabel("Camera:"))
        vis_controls.addWidget(self.view_combo)
        vis_controls.addStretch()
        vis_controls.addWidget(btn_reset_view)
        vis_layout.addLayout(vis_controls)

        # -------------------------------------------------------------
        # THE NEW OPENGL CANVAS
        # -------------------------------------------------------------
        self.gl_canvas = gl.GLViewWidget()
        self.gl_canvas.setBackgroundColor('#2b2b2b')

        # Add a subtle grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=500, y=500, z=0)
        self.grid.setSpacing(x=10, y=10, z=10)
        self.grid.setColor((100, 100, 100, 100)) # subtle gray
        self.gl_canvas.addItem(self.grid)

        # Add Axis
        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=20, y=20, z=20)
        self.gl_canvas.addItem(self.axis)

        # References to plot items
        self.line_item = None
        self.scatter_item = None
        self.start_point = None
        self.end_point = None

        vis_layout.addWidget(self.gl_canvas)
        
        visualizer_group.setLayout(vis_layout)
        return visualizer_group

    # =========================================================================
    # OPENGL PLOTTING LOGIC
    # =========================================================================
    def set_camera_view(self) -> None:
        """Adjusts the OpenGL camera angle instantly based on the ComboBox."""
        view_mode = self.view_combo.currentText()
        distance = self.gl_canvas.opts['distance'] # preserve zoom
        
        if view_mode == "Top View (XY)":
            self.gl_canvas.setCameraPosition(elevation=90, azimuth=-90, distance=distance)
        elif view_mode == "Side View (YZ)":
            self.gl_canvas.setCameraPosition(elevation=0, azimuth=0, distance=distance)
        elif view_mode == "Front View (XZ)":
            self.gl_canvas.setCameraPosition(elevation=0, azimuth=-90, distance=distance)
        else:
            # Default 3D
            self.gl_canvas.setCameraPosition(elevation=30, azimuth=-45, distance=distance)

    def update_plot(self) -> None:
        """Extracts table data and updates the hardware-accelerated OpenGL plot."""
        points = []
        for row in range(self.table_waypoints.rowCount()):
            try:
                x_i = self.table_waypoints.item(row, 0)
                y_i = self.table_waypoints.item(row, 1)
                z_i = self.table_waypoints.item(row, 2)
                if x_i and y_i and z_i:
                    points.append([float(x_i.text()), float(y_i.text()), -float(z_i.text())])
            except ValueError:
                pass

        # Clean old plot items
        if self.line_item: self.gl_canvas.removeItem(self.line_item)
        if self.scatter_item: self.gl_canvas.removeItem(self.scatter_item)
        if self.start_point: self.gl_canvas.removeItem(self.start_point)
        if self.end_point: self.gl_canvas.removeItem(self.end_point)

        if not points:
            return

        # Convert to numpy array for OpenGL
        pts_array = np.array(points, dtype=np.float32)

        # 1. Main Path Line
        self.line_item = gl.GLLinePlotItem(pos=pts_array, color=pg.glColor('#00d4ff'), width=2.0)
        self.gl_canvas.addItem(self.line_item)

        # 2. General Points (White)
        self.scatter_item = gl.GLScatterPlotItem(pos=pts_array, color=pg.glColor('w'), size=8.0)
        self.gl_canvas.addItem(self.scatter_item)

        # 3. Start Point (Green)
        self.start_point = gl.GLScatterPlotItem(pos=np.array([pts_array[0]]), color=pg.glColor('#4caf50'), size=15.0)
        self.gl_canvas.addItem(self.start_point)

        # 4. End Point (Red)
        if len(pts_array) > 1:
            self.end_point = gl.GLScatterPlotItem(pos=np.array([pts_array[-1]]), color=pg.glColor('#f44336'), size=15.0)
            self.gl_canvas.addItem(self.end_point)

    # =========================================================================
    # MATHEMATICAL ENGINE INTEGRATION AND GENERAL LOGIC
    # =========================================================================
    def start_simulation(self) -> None:
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
        self.lbl_sim_status.setStyleSheet("color: #00ff00; font-size: 13px; font-weight: bold;")

    def _collect_aircraft_data(self) -> Dict[str, float]:
        data = {}
        tab_ac = getattr(self.main_window, "tab_aircraft", None)
        if tab_ac:
            for key, spin in tab_ac.aircraft_spins.items():
                data[key] = spin.value()
        return data

    def _collect_mission_data(self) -> Dict[str, Any]:
        data = {
            "coordinate_system": "cartesian" if self.radio_cartesian.isChecked() else "geodesic",
            "wind": {
                "use_api": self.radio_api_wind.isChecked(),
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
            self.radio_manual_wind.setChecked(True)
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
            if wind_data.get("use_api", False):
                self.radio_api_wind.setChecked(True)
            else:
                self.radio_manual_wind.setChecked(True)
                
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
                self._add_row_to_table(wp.get("x", 0.0), wp.get("y", 0.0), wp.get("z", 0.0))

            traj_data = data.get("trajectory", {})
            self.combo_traj_type.setCurrentText(traj_data.get("type", "5th Degree Polynomial"))
            self.spin_speed_xy.setValue(traj_data.get("speed_xy", 10.0))
            self.spin_speed_z.setValue(traj_data.get("speed_z", 5.0))
            self.combo_yaw_mode.setCurrentText(traj_data.get("yaw_mode", "Free (Initial Fixed)"))

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
        is_api = getattr(self, 'radio_api_wind', None) and self.radio_api_wind.isChecked()
        
        if is_api:
            self.spin_wind_gust.setEnabled(True) 
            self.spin_wind_gust.setReadOnly(True)
        else:
            self.spin_wind_gust.setReadOnly(False)
            wind_type = self.combo_wind_type.currentText()
            if "None" in wind_type or "Constant" in wind_type:
                self.spin_wind_gust.setEnabled(False)
                self.spin_wind_gust.setValue(0.0)
            else:
                self.spin_wind_gust.setEnabled(True)

    def _toggle_weather_mode(self) -> None:
        is_api = self.radio_api_wind.isChecked()
        self.loc_container.setVisible(is_api)
        
        self.combo_wind_type.setEnabled(not is_api)
        
        spins = [
            self.spin_wind_mag,
            self.spin_wind_head,
            self.spin_wind_elev,
            self.spin_wind_gust
        ]
        
        self.toggle_gust_input()
        
        if is_api:
            locked_style = (
                "QDoubleSpinBox {"
                "background-color: #1a1a1a; "
                "color: #00d4ff; "
                "border: 1px dashed #555555; "
                "border-radius: 4px; "
                "padding: 2px;"
                "}"
            )
            for spin in spins:
                spin.setReadOnly(True)
                spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spin.setStyleSheet(locked_style)
        else:
            for spin in spins:
                spin.setReadOnly(False)
                spin.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
                spin.setStyleSheet("")
                
        self.toggle_gust_input()

    def fetch_weather_data(self) -> None:
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
        selected_rows = set(item.row() for item in self.table_waypoints.selectedItems())
        count = len(selected_rows)
        
        self.btn_edit.setEnabled(count == 1)
        self.btn_up.setEnabled(count == 1)
        self.btn_down.setEnabled(count == 1)
        self.btn_dup.setEnabled(count >= 1)
        self.btn_rem.setEnabled(count >= 1)

    def handle_item_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if self._last_selected_row == row:
            self.table_waypoints.clearSelection()
            self._last_selected_row = -1
        else:
            self._last_selected_row = row

    def _add_row_to_table(self, x: float, y: float, z: float) -> None:
        row = self.table_waypoints.rowCount()
        self.table_waypoints.insertRow(row)
        dec = 2 if self.radio_cartesian.isChecked() else 6
        
        self.table_waypoints.setItem(row, 0, QTableWidgetItem(f"{x:.{dec}f}"))
        self.table_waypoints.setItem(row, 1, QTableWidgetItem(f"{y:.{dec}f}"))
        self.table_waypoints.setItem(row, 2, QTableWidgetItem(f"{z:.2f}"))

    def add_manual_waypoint(self) -> None:
        self._add_row_to_table(
            self.spin_x.value(), self.spin_y.value(), self.spin_z.value()
        )
        self.spin_x.setValue(0.0)
        self.spin_y.setValue(0.0)
        self.spin_z.setValue(0.0)

    def edit_waypoint(self) -> None:
        current_row = self.table_waypoints.currentRow()
        if current_row >= 0:
            self.table_waypoints.editItem(self.table_waypoints.item(current_row, 0))

    def remove_waypoint(self) -> None:
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
        current_row = self.table_waypoints.currentRow()
        if current_row > 0:
            self._swap_rows(current_row, current_row - 1)
            self.table_waypoints.setCurrentCell(current_row - 1, 0)

    def move_waypoint_down(self) -> None:
        current_row = self.table_waypoints.currentRow()
        if current_row >= 0 and current_row < self.table_waypoints.rowCount() - 1:
            self._swap_rows(current_row, current_row + 1)
            self.table_waypoints.setCurrentCell(current_row + 1, 0)

    def _swap_rows(self, row1: int, row2: int) -> None:
        for col in range(self.table_waypoints.columnCount()):
            item1 = self.table_waypoints.takeItem(row1, col)
            item2 = self.table_waypoints.takeItem(row2, col)
            self.table_waypoints.setItem(row1, col, item2)
            self.table_waypoints.setItem(row2, col, item1)

    def load_waypoints_from_file(self) -> None:
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
        self.set_camera_view()

    def toggle_yaw_target(self) -> None:
        is_target_mode = "Target" in self.combo_yaw_mode.currentText()
        self.yaw_target_container.setVisible(is_target_mode)
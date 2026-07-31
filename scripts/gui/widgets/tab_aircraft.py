"""Aircraft configuration UI module for the UAV simulator.

Provides the graphical interface for setting physical, aerodynamic, and 
propulsion parameters of the UAV, saving them as JSON profiles, and 
exporting them to ROS 2 YAML configuration files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# =========================================================================
# PATH CONFIGURATION
# =========================================================================
# 1. Get the current script directory (src/gui/widgets)
current_dir = Path(__file__).resolve().parent

# 2. Go up two levels to reach the 'src' folder
src_dir = current_dir.parent.parent

# 3. Add 'src' to sys.path so Python can locate paths.py
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from paths import AIRCRAFT_MODELS_DIR, ROS_CONFIG_DIR, ROS_INSTALL_DIR


# =========================================================================
# UTILITY CLASSES (UI FACTORY AND ALERTS)
# =========================================================================
class UIFactory:
    """Factory class for generating standard UI components."""

    @staticmethod
    def create_toolbar_button(
        style: QStyle, icon_enum: QStyle.StandardPixmap, tooltip: str
    ) -> QPushButton:
        """Creates a standardized toolbar button."""
        btn = QPushButton()
        btn.setIcon(style.standardIcon(icon_enum))
        btn.setToolTip(tooltip)
        return btn

    @staticmethod
    def create_param_group(title: str) -> Tuple[QGroupBox, QFormLayout]:
        """Creates a standardized group box with a form layout."""
        grp = QGroupBox(title)
        grp.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        form = QFormLayout(grp)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        form.setHorizontalSpacing(15)
        form.setVerticalSpacing(10)
        return grp, form

    @staticmethod
    def create_spinbox(decimals: int = 3, min_val: float = 0.0, max_val: float = 99999.0) -> QDoubleSpinBox:
        """Creates a standardized double spinbox for parameter input.
        
        Args:
            decimals (int): The number of decimal places to display.
            min_val (float): Minimum allowed value (prevents physical impossibilities).
            max_val (float): Maximum allowed value.
        """
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setValue(0.0)
        spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)

        spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        spin.setMinimumHeight(28)
        spin.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return spin


class ParameterHelpDialog(QDialog):
    """Floating help dialog detailing the UAV parameters and UX guide."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the help dialog and its HTML content."""
        super().__init__(parent)
        self.setWindowTitle("Documentation - UAV Parameters")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        # Gets the absolute path for local assets (future-proofing for GIFs)
        assets_dir = current_dir.parent.parent / "assets" / "img"
        assets_path_str = assets_dir.as_posix()

        html_content = f"""
        <html>
        <head>
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 14px; 
                line-height: 1.6; 
                color: #e0e0e0; 
                padding: 15px 25px; 
            }}
            h2 {{ color: #00d4ff; margin-top: 10px; border-bottom: 1px solid #555; 
                 padding-bottom: 5px;}}
            h3 {{ color: #4caf50; margin-top: 25px; margin-bottom: 5px; }}
            ul {{ margin-top: 5px; padding-left: 25px; }}
            li {{ margin-bottom: 8px; }}
            b {{ color: #ffffff; }}
            .usage-box {{
                background-color: #353535;
                border-left: 4px solid #00d4ff;
                padding: 10px 15px;
                margin-top: 15px;
            }}
            .img-placeholder {{
                display: block;
                margin: 20px auto;
                max-width: 80%;
                border: 1px solid #555;
            }}
        </style>
        </head>
        <body>
        <h2>UAV Configuration & Usage Guide</h2>
        
        <div class="usage-box">
            <b>Quick Start:</b>
            <ul>
                <li><b>Load Model (Ctrl+O):</b> Loads a previously saved `.json` profile.</li>
                <li><b>Save Model (Ctrl+S):</b> Saves your current parameters and automatically updates the ROS 2 physical engine (`aircraft_params.yaml`).</li>
                <li><b>Reset (Ctrl+R):</b> Clears all fields safely.</li>
            </ul>
            <i>Note: Physical properties like Mass and Inertia are strictly prevented from taking negative values to maintain mathematical stability.</i>
        </div>

        <!-- Placeholder for future GIF/Media integration -->
        <!-- <img class="img-placeholder" src="file:///{assets_path_str}/aircraft_tutorial.gif" alt="Tutorial GIF"> -->

        <h2>Dynamic Simulation Parameters Glossary</h2>
        <p>This section details the physical attributes required to simulate the multirotor's behavior with high fidelity.</p>
        
        <h3>1. Mass and Inertia</h3>
        <ul>
            <li><b>mass:</b> Total drone mass (mB) [kg].</li>
            <li><b>jx:</b> Moment of inertia on the x-axis [kg&middot;m&sup2;].</li>
            <li><b>jy:</b> Moment of inertia on the y-axis [kg&middot;m&sup2;].</li>
            <li><b>jz:</b> Moment of inertia on the z-axis [kg&middot;m&sup2;].</li>
            <li><b>jxz:</b> Product of inertia in the xz-plane [kg&middot;m&sup2;]. <i>(Can be negative)</i></li>
        </ul>

        <h3>2. Center of Mass Distances</h3>
        <ul>
            <li><b>dx_arm:</b> Distance from the CM to the motor on the x-axis [m].</li>
            <li><b>dy_fw:</b> Distance from the CM to the front motor on the y-axis [m].</li>
            <li><b>dy_bw:</b> Distance from the CM to the rear motor on the y-axis [m].</li>
            <li><b>dz:</b> Distance from the CM to the motor on the z-axis [m].</li>
        </ul>
        
        <h3>3. Propulsion (Motor & Propeller)</h3>
        <ul>
            <li><b>tau (&tau;):</b> Motor time constant (1st order dynamics) [s].</li>
            <li><b>kp (K<sub>p</sub>):</b> Motor correction gain [dimensionless].</li>
            <li><b>xi (&xi;):</b> Motor damping coefficient [dimensionless].</li>
            <li><b>omega_min:</b> Minimum motor rotation speed [rad/s].</li>
            <li><b>omega_max:</b> Maximum motor rotation speed [rad/s].</li>
            <li><b>prop_diameter:</b> Propeller physical diameter [m].</li>
            <li><b>c_t0 (C<sub>T0</sub>):</b> Propeller static thrust coef. [dimensionless].</li>
            <li><b>c_p0 (C<sub>P0</sub>):</b> Propeller static power coef. [dimensionless].</li>
            <li><b>c_q0 (C<sub>Q0</sub>):</b> Propeller static moment coef. [dimensionless].</li>
        </ul>

        <h3>4. Global Aerodynamics</h3>
        <ul>
            <li><b>cd (C<sub>d</sub>):</b> Aerodynamic drag coefficient [dimensionless].</li>
            <li><b>k_t0 (k<sub>T0</sub>):</b> Thrust coefficient constant [N/(rad/s)&sup2;].</li>
            <li><b>k_q0 (k<sub>Q0</sub>):</b> Torque coefficient constant [N&middot;m/(rad/s)&sup2;].</li>
        </ul>

        <h3>5. Kinematic Limits (Safety)</h3>
        <ul>
            <li><b>max_horizontal_speed:</b> Maximum horizontal linear speed [m/s].</li>
            <li><b>cruise_speed:</b> Nominal horizontal linear speed [m/s].</li>
            <li><b>max_ascent_speed:</b> Maximum vertical linear speed during climb [m/s]. <i>(Usually negative in NED frame)</i></li>
            <li><b>max_descent_speed:</b> Maximum vertical linear speed during fall [m/s].</li>
            <li><b>max_tilt_angle:</b> Maximum Pitch/Roll angle [rad].</li>
            <li><b>max_roll_pitch_rate:</b> Maximum angular velocity rate (Roll/Pitch) [rad/s].</li>
            <li><b>max_yaw_rate:</b> Maximum angular velocity rate on the Yaw axis [rad/s].</li>
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
class TabAircraft(QWidget):
    """UI Tab for managing UAV model parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.aircraft_spins: Dict[str, QDoubleSpinBox] = {}
        self._build_ui()
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_new.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_new.activated.connect(self._trigger_reset)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_save.activated.connect(self._trigger_save)

        self.shortcut_load = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_load.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_load.activated.connect(self._trigger_load)

        self.shortcut_help = QShortcut(QKeySequence("F1"), self)
        self.shortcut_help.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_help.activated.connect(self._trigger_help)

    def _trigger_reset(self) -> None:
        if self.isVisible():
            self.reset_aircraft()

    def _trigger_save(self) -> None:
        if self.isVisible():
            self.save_aircraft()

    def _trigger_load(self) -> None:
        if self.isVisible():
            self.load_aircraft_file()

    def _trigger_help(self) -> None:
        if self.isVisible():
            self.show_help_window()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        layout.addLayout(self._build_top_section())
        layout.addLayout(self._build_bottom_section())
        layout.addStretch()

    def _build_top_section(self) -> QVBoxLayout:
        top_layout = QVBoxLayout()
        top_layout.setSpacing(15)

        global_toolbar_layout = QHBoxLayout()
        style = self.style()

        btn_new = UIFactory.create_toolbar_button(
            style, QStyle.StandardPixmap.SP_FileIcon, "New Model (Ctrl+R)"
        )
        btn_new.clicked.connect(self.reset_aircraft)

        btn_save = UIFactory.create_toolbar_button(
            style, QStyle.StandardPixmap.SP_DriveFDIcon, "Save Current Model (Ctrl+S)"
        )
        btn_save.clicked.connect(self.save_aircraft)

        btn_load = UIFactory.create_toolbar_button(
            style, QStyle.StandardPixmap.SP_DirIcon, "Load Saved Model (Ctrl+O)"
        )
        btn_load.clicked.connect(self.load_aircraft_file)

        global_toolbar_layout.addWidget(btn_new)
        global_toolbar_layout.addWidget(btn_save)
        global_toolbar_layout.addWidget(btn_load)

        global_toolbar_layout.addStretch()

        btn_help = UIFactory.create_toolbar_button(
            style,
            QStyle.StandardPixmap.SP_MessageBoxInformation,
            "Parameter Details & Guide (F1)",
        )
        btn_help.clicked.connect(self.show_help_window)
        global_toolbar_layout.addWidget(btn_help)

        top_layout.addLayout(global_toolbar_layout)

        grp_id, form_id = UIFactory.create_param_group("Identification")
        lbl_id = QLabel("Model Name:")
        lbl_id.setMinimumWidth(150)

        self.line_aircraft_name = QLineEdit()
        self.line_aircraft_name.setPlaceholderText("e.g.: DJI Matrice 350 RTK")
        self.line_aircraft_name.setMinimumHeight(28)
        self.line_aircraft_name.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.line_aircraft_name.setStyleSheet(
            "QLineEdit { color: #ffffff; background-color: #2b2b2b; "
            "border: 1px solid #555555; padding-left: 5px; }\n"
            "QLineEdit::placeholder { color: #aaaaaa; }"
        )

        form_id.addRow(lbl_id, self.line_aircraft_name)
        top_layout.addWidget(grp_id)

        return top_layout

    def show_help_window(self) -> None:
        self.help_dialog = ParameterHelpDialog(self)
        self.help_dialog.show()

    def _build_bottom_section(self) -> QHBoxLayout:
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        # Tuple structure: (Label, Decimals, Min_Value)
        col1_data = [
            (
                "Mass and Inertia",
                {
                    "mass": ("Mass (kg):", 3, 0.0),
                    "jx": ("Inertia X (kg·m²):", 3, 0.0),
                    "jy": ("Inertia Y (kg·m²):", 3, 0.0),
                    "jz": ("Inertia Z (kg·m²):", 3, 0.0),
                    "jxz": ("Inertia Product XZ:", 3, -999.0), # Product of inertia can be negative
                },
            ),
            (
                "Center of Mass Distances",
                {
                    "dx_arm": ("X-Axis - dx (m):", 3, 0.0),
                    "dy_fw": ("Y-Axis - Front (m):", 3, 0.0),
                    "dy_bw": ("Y-Axis - Rear (m):", 3, 0.0),
                    "dz": ("Z-Axis - dz (m):", 3, 0.0),
                },
            ),
        ]

        col2_data = [
            (
                "Propulsion (Motor & Propeller)",
                {
                    "tau": ("Motor Constant (τ):", 3, 0.0),
                    "kp": ("Correction Gain (K<sub>p</sub>):", 3, 0.0),
                    "xi": ("Damping (ξ):", 3, 0.0),
                    "omega_min": ("Min Rotation (rad/s):", 3, 0.0),
                    "omega_max": ("Max Rotation (rad/s):", 3, 0.0),
                    "prop_diameter": ("Prop. Diameter (m):", 4, 0.0),
                    "c_t0": ("Thrust Coef (C<sub>T0</sub>):", 4, 0.0),
                    "c_p0": ("Power Coef (C<sub>P0</sub>):", 4, 0.0),
                    "c_q0": ("Torque Coef (C<sub>Q0</sub>):", 4, 0.0),
                },
            ),
            (
                "Global Aerodynamics",
                {
                    "cd": ("Drag Coef (C<sub>d</sub>):", 3, 0.0),
                    "k_t0": ("Thrust Const (k<sub>T0</sub>):", 8, 0.0),
                    "k_q0": ("Torque Const (k<sub>Q0</sub>):", 8, 0.0),
                },
            ),
        ]

        col3_data = [
            (
                "Kinematic Limits (Safety)",
                {
                    "max_horizontal_speed": ("Max Horiz. Speed (m/s):", 3, 0.0),
                    "cruise_speed": ("Cruise Speed (m/s):", 3, 0.0),
                    "max_ascent_speed": ("Max Ascent Speed (m/s):", 3, -999.0), # Negative in NED
                    "max_descent_speed": ("Max Descent Speed (m/s):", 3, 0.0),
                    "max_tilt_angle": ("Max Tilt Angle (rad):", 3, 0.0),
                    "max_roll_pitch_rate": ("Max Roll/Pitch Rate:", 3, 0.0),
                    "max_yaw_rate": ("Max Yaw Rate (rad/s):", 3, 0.0),
                },
            )
        ]

        bottom_layout.addLayout(self._create_column_layout(col1_data))
        bottom_layout.addLayout(self._create_column_layout(col2_data))
        bottom_layout.addLayout(self._create_column_layout(col3_data))

        return bottom_layout

    def _create_column_layout(
        self, groups_data: List[Tuple[str, Dict[str, Tuple[str, int, float]]]]
    ) -> QVBoxLayout:
        col_layout = QVBoxLayout()
        for title, params in groups_data:
            grp, form = UIFactory.create_param_group(title)
            for key, (label_text, decimals, min_val) in params.items():
                spin = UIFactory.create_spinbox(decimals, min_val=min_val)
                self.aircraft_spins[key] = spin

                lbl = QLabel(label_text)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setMinimumWidth(160)

                font = lbl.font()
                font.setPointSize(10)
                lbl.setFont(font)
                spin.setFont(font)

                form.addRow(lbl, spin)

            col_layout.addWidget(grp)
        col_layout.addStretch()
        return col_layout

    def reset_aircraft(self) -> None:
        msg = CustomMessageBox(
            "New Model",
            "Do you want to reset all parameters?",
            "Unsaved data will be lost.",
            msg_type="question",
            parent=self,
        )
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.line_aircraft_name.clear()
            for spin in self.aircraft_spins.values():
                spin.setValue(0.0)

    def save_aircraft(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Aircraft Model", str(AIRCRAFT_MODELS_DIR), "JSON Files (*.json)"
        )
        if not file_path:
            return

        if not file_path.endswith(".json"):
            file_path += ".json"

        aircraft_data: Dict[str, Any] = {"name": self.line_aircraft_name.text()}
        for key, spin in self.aircraft_spins.items():
            aircraft_data[key] = spin.value()

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(aircraft_data, file, indent=4)

            self.export_to_ros_yaml()

            msg = CustomMessageBox(
                "Success",
                "Model saved successfully!",
                "JSON saved and ROS 2 YAML updated.",
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
        ros_data = {
            "quadcopter_node": {
                "ros__parameters": {
                    "mass": self.aircraft_spins["mass"].value(),
                    "jx": self.aircraft_spins["jx"].value(),
                    "jy": self.aircraft_spins["jy"].value(),
                    "jz": self.aircraft_spins["jz"].value(),
                    "jxz": self.aircraft_spins["jxz"].value(),
                    "cd": self.aircraft_spins["cd"].value(),
                    "k_t0": self.aircraft_spins["k_t0"].value(),
                    "k_q0": self.aircraft_spins["k_q0"].value(),
                    "tau": self.aircraft_spins["tau"].value(),
                    "kp": self.aircraft_spins["kp"].value(),
                    "xi": self.aircraft_spins["xi"].value(),
                    "dx_arm": self.aircraft_spins["dx_arm"].value(),
                    "dy_fw": self.aircraft_spins["dy_fw"].value(),
                    "dy_bw": self.aircraft_spins["dy_bw"].value(),
                    "omega_min": self.aircraft_spins["omega_min"].value(),
                    "omega_max": self.aircraft_spins["omega_max"].value(),
                }
            },
            "controller_node": {
                "ros__parameters": {
                    "mass": self.aircraft_spins["mass"].value(),
                    "max_ascent_speed": self.aircraft_spins["max_ascent_speed"].value(),
                    "max_descent_speed": self.aircraft_spins["max_descent_speed"].value(),
                    "cruise_speed": self.aircraft_spins["cruise_speed"].value(),
                    "max_tilt_angle": self.aircraft_spins["max_tilt_angle"].value(),
                    "max_roll_pitch_rate": self.aircraft_spins["max_roll_pitch_rate"].value(),
                    "max_yaw_rate": self.aircraft_spins["max_yaw_rate"].value(),
                }
            },
        }

        yaml_path_src = ROS_CONFIG_DIR / "aircraft_params.yaml"
        with open(yaml_path_src, "w", encoding="utf-8") as file:
            yaml.dump(ros_data, file, default_flow_style=False, sort_keys=False)

        if ROS_INSTALL_DIR and ROS_INSTALL_DIR.exists():
            yaml_path_install = ROS_INSTALL_DIR / "aircraft_params.yaml"
            with open(yaml_path_install, "w", encoding="utf-8") as file:
                yaml.dump(ros_data, file, default_flow_style=False, sort_keys=False)

    def load_aircraft_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Aircraft", str(AIRCRAFT_MODELS_DIR), "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if "name" in data:
                self.line_aircraft_name.setText(data["name"])
            else:
                self.line_aircraft_name.clear()

            for key, value in data.items():
                if key in self.aircraft_spins:
                    self.aircraft_spins[key].setValue(float(value))

            filename = os.path.basename(file_path)
            msg = CustomMessageBox(
                "Success",
                "Parameters loaded!",
                f"Model '{filename}' loaded successfully.",
                msg_type="success",
                parent=self,
            )
            msg.exec()

        except Exception as err:
            msg = CustomMessageBox(
                "Load Error", "Could not load the parameters.", str(err), 
                msg_type="error", parent=self
            )
            msg.exec()
"""Live telemetry visualization and ground control station module.

This module provides interactive PyQtGraph-based dashboards for monitoring 
UAV states (position, attitude, velocities) in real-time. It supports data 
history buffering, CSV exporting, and off-screen high-resolution rendering.
"""

import sys
import threading
import http.server
import socketserver
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStyle,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QStackedWidget,
)

# IMPORTAÇÃO DO NAVEGADOR EMBUTIDO
from PySide6.QtWebEngineWidgets import QWebEngineView

# =========================================================================
# INTEGRAÇÃO COM O PATHS.PY
# =========================================================================
_current_path = Path(__file__).resolve().parent
for _ in range(6):
    if (_current_path / "package.xml").exists():
        sys.path.insert(0, str(_current_path))
        break
    _current_path = _current_path.parent
else:
    sys.path.insert(0, "/workspace/src/m300_sim")

import paths

# =========================================================================
# DIALOG BOXES
# =========================================================================
class TelemetryHelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Documentation - Live Telemetry")
        self.resize(800, 500)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        
        html_content = (
            "<html><head><style>"
            "body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #e0e0e0; padding: 15px 25px; }"
            "h2 { color: #00d4ff; margin-top: 10px; border-bottom: 1px solid #555; padding-bottom: 5px;}"
            "h3 { color: #4caf50; margin-top: 25px; margin-bottom: 5px; }"
            "ul { margin-top: 5px; padding-left: 25px; }"
            "li { margin-bottom: 8px; }"
            "b { color: #ffffff; }"
            "</style></head><body>"
            "<h2>Live Telemetry Dashboard Guide</h2>"
            "<p>This panel acts as the Ground Control Station (GCS) for your UAV, monitoring data streamed directly from the ROS 2 environment in real-time.</p>"
            "<h3>1. Interactive Graphs & History</h3><ul>"
            "<li><b>Pan & Zoom:</b> The graphs are fully interactive. Use the mouse wheel to zoom in/out, and click-drag to pan across the timeline.</li>"
            "<li><b>Auto-Scroll Toggle:</b> Uncheck the 'Auto-Scroll' box to stop the live window and view the complete flight history.</li>"
            "<li><b>Switching Views:</b> Cycle between data categories, including the live 3D Digital Twin environment.</li>"
            "</ul><h3>2. Data Recording & Export</h3><ul>"
            "<li><b>Export CSV:</b> Generates a complete mathematical report of the flight.</li>"
            "<li><b>Snapshot Graph:</b> Render a high-res PNG off-screen for reports.</li>"
            "<li><b>Clear History:</b> Wipes the background memory and resets the dashboard timeline.</li>"
            "</ul></body></html>"
        )
        browser.setHtml(html_content)
        layout.addWidget(browser)

class CustomMessageBox(QDialog):
    def __init__(self, title: str, main_text: str, detail_text: str = "", msg_type: str = "info", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; } "
            "QLabel { color: #e0e0e0; font-size: 14px;} "
            "QPushButton { background-color: #0d6efd; color: white; border-radius: 4px; padding: 6px 16px; font-weight: bold; min-width: 80px; } "
            "QPushButton:hover { background-color: #0b5ed7; }"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel(f"<h3 style='margin: 0;'>{main_text}</h3>"))
        if detail_text:
            layout.addWidget(QLabel(detail_text))
            
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

class ExportGraphDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Export Options")
        self.setMinimumWidth(400)
        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; color: white; } "
            "QLabel { color: #e0e0e0; font-size: 14px;} "
            "QPushButton { background-color: #0d6efd; color: white; border-radius: 4px; padding: 6px 16px; font-weight: bold; } "
            "QComboBox { background-color: #3b3b3b; color: white; padding: 5px; border: 1px solid #555; font-size: 13px;}"
        )
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the specific data group to render and export:"))
        
        self.combo = QComboBox()
        self.options = [
            ("Current Visible Screen (WYSIWYG)", []),
            ("Full Dashboard Overview (12 Plots)", ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']),
            ("Position - All (North, East, Alt)", ['x', 'y', 'z']),
            ("Position - Specific: North", ['x']),
            ("Position - Specific: East", ['y']),
            ("Position - Specific: Altitude", ['z']),
            ("Attitude - All (Roll, Pitch, Yaw)", ['roll', 'pitch', 'yaw']),
            ("Attitude - Specific: Roll", ['roll']),
            ("Attitude - Specific: Pitch", ['pitch']),
            ("Attitude - Specific: Yaw", ['yaw']),
            ("Linear Velocity - All (u, v, w)", ['u', 'v', 'w']),
            ("Linear Velocity - Specific: u", ['u']),
            ("Linear Velocity - Specific: v", ['v']),
            ("Linear Velocity - Specific: w", ['w']),
            ("Angular Rate - All (p, q, r)", ['p', 'q', 'r']),
            ("Angular Rate - Specific: p", ['p']),
            ("Angular Rate - Specific: q", ['q']),
            ("Angular Rate - Specific: r", ['r'])
        ]
        for text, _ in self.options:
            self.combo.addItem(text)
            
        layout.addWidget(self.combo)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_export = QPushButton("Render & Save PNG")
        self.btn_export.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)
        self.selected_index = 0

    def accept(self) -> None:
        self.selected_index = self.combo.currentIndex()
        super().accept()

pg.setConfigOption('background', '#2b2b2b')
pg.setConfigOption('foreground', '#aaaaaa')
pg.setConfigOptions(antialias=True) 

# =========================================================================
# MAIN TAB CLASS
# =========================================================================
class TabTelemetry(QWidget):
    def __init__(self, main_window_ref: QWidget) -> None:
        super().__init__()
        self.main_window = main_window_ref 
        
        self.window_size = 300 
        self.online_start_t: Optional[float] = None
        self.webgl_port = 8000
        
        self._init_rolling_buffer()
        self.flight_history = self._create_empty_history()
        self.graph_items: List[tuple] = [] 
        
        self._start_webgl_server()
        
        self._setup_dark_theme()
        self._build_ui()
        self._setup_shortcuts()
        self.init_graphs()

    def _start_webgl_server(self) -> None:
        webgl_dir = paths.PROJECT_ROOT / "webgl"
        
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(webgl_dir), **kwargs)
            def log_message(self, format, *args):
                pass
                
        try:
            socketserver.TCPServer.allow_reuse_address = True
            self.httpd = socketserver.TCPServer(("127.0.0.1", self.webgl_port), QuietHandler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"[GUI] Servidor WebGL interno iniciado na porta {self.webgl_port} (Diretório: {webgl_dir})")
        except Exception as e:
            print(f"[GUI Web Server] Porta em uso: {e}")

    def _init_rolling_buffer(self) -> None:
        t_array = np.linspace(-10.0, 0.0, self.window_size)
        self.online_data: Dict[str, np.ndarray] = {
            't': t_array.copy(), 'x': np.zeros(self.window_size), 'y': np.zeros(self.window_size), 'z': np.zeros(self.window_size),
            'roll': np.zeros(self.window_size), 'pitch': np.zeros(self.window_size), 'yaw': np.zeros(self.window_size),
            'u': np.zeros(self.window_size), 'v': np.zeros(self.window_size), 'w': np.zeros(self.window_size),
            'p': np.zeros(self.window_size), 'q': np.zeros(self.window_size), 'r': np.zeros(self.window_size)
        }

    def _create_empty_history(self) -> Dict[str, List[float]]:
        return {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 'u': [], 'v': [], 'w': [], 'p': [], 'q': [], 'r': []}

    def _setup_dark_theme(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        self.setPalette(palette)
        self.setStyleSheet(
            "QGroupBox { border: 1px solid #555555; border-radius: 4px; margin-top: 2ex; padding-top: 10px; color: #e0e0e0; font-weight: bold;} "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #00d4ff; } "
            "QPushButton { background-color: #3b3b3b; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; } "
            "QPushButton:hover { background-color: #0d6efd; border: 1px solid #0d6efd; } "
            "QCheckBox { color: white; font-weight: bold; } "
            "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; background: #2b2b2b; } "
            "QCheckBox::indicator:checked { background: #00d4ff; border: 1px solid #00d4ff; }"
        )

    def _setup_shortcuts(self) -> None:
        self.shortcut_cycle = QShortcut(QKeySequence("Tab"), self)
        self.shortcut_cycle.activated.connect(self.cycle_views)
        self.shortcut_help = QShortcut(QKeySequence("F1"), self)
        self.shortcut_help.activated.connect(self.show_help_window)

    def cycle_views(self) -> None:
        current_id = self.radio_group.checkedId()
        next_id = (current_id + 1) % len(self.radios)
        self.radios[next_id].setChecked(True)
        self.init_graphs()

    def show_help_window(self) -> None:
        self.help_dialog = TelemetryHelpDialog(self)
        self.help_dialog.show()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        toolbar_layout = QHBoxLayout()
        self.lbl_live_status = QLabel("● SYSTEM STANDBY")
        self.lbl_live_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 14px;")
        toolbar_layout.addWidget(self.lbl_live_status)
        toolbar_layout.addStretch()
        
        self.btn_clear = QPushButton(" Clear History")
        self.btn_clear.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_clear.clicked.connect(self.clear_history)
        
        self.btn_export_img = QPushButton(" Snapshot Graph")
        self.btn_export_img.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.btn_export_img.clicked.connect(self.export_graph_image)
        
        self.btn_export_csv = QPushButton(" Export Flight CSV")
        self.btn_export_csv.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon))
        self.btn_export_csv.clicked.connect(self.export_csv)
        
        self.btn_help = QPushButton(" Help")
        self.btn_help.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.btn_help.clicked.connect(self.show_help_window)
        
        toolbar_layout.addWidget(self.btn_clear)
        toolbar_layout.addWidget(self.btn_export_img)
        toolbar_layout.addWidget(self.btn_export_csv)
        toolbar_layout.addWidget(self.btn_help)
        main_layout.addLayout(toolbar_layout)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        
        dash_panel = QFrame()
        dash_panel.setFixedWidth(280)
        dash_layout = QVBoxLayout(dash_panel)
        dash_layout.setContentsMargins(0, 0, 0, 0)
        dash_layout.setSpacing(10)

        time_grp = QGroupBox("Mission Time")
        time_lay = QVBoxLayout(time_grp)
        self.lbl_time = self._create_hud_value("0.00 s", color="#00ff00", size=24)
        time_lay.addWidget(self.lbl_time)
        dash_layout.addWidget(time_grp)

        pos_grp = QGroupBox("Global Position [m]")
        pos_lay = QGridLayout(pos_grp)
        self.hud_n = self._add_hud_row(pos_lay, 0, "North (X):")
        self.hud_e = self._add_hud_row(pos_lay, 1, "East (Y):")
        self.hud_d = self._add_hud_row(pos_lay, 2, "Altitude (-Z):")
        dash_layout.addWidget(pos_grp)

        att_grp = QGroupBox("Attitude [deg]")
        att_lay = QGridLayout(att_grp)
        self.hud_roll = self._add_hud_row(att_lay, 0, "Roll:")
        self.hud_pitch = self._add_hud_row(att_lay, 1, "Pitch:")
        self.hud_yaw = self._add_hud_row(att_lay, 2, "Yaw:")
        dash_layout.addWidget(att_grp)

        lin_grp = QGroupBox("Linear Velocity [m/s]")
        lin_lay = QGridLayout(lin_grp)
        self.hud_u = self._add_hud_row(lin_lay, 0, "Vel u:")
        self.hud_v = self._add_hud_row(lin_lay, 1, "Vel v:")
        self.hud_w = self._add_hud_row(lin_lay, 2, "Vel w:")
        dash_layout.addWidget(lin_grp)

        ang_grp = QGroupBox("Angular Rate [rad/s]")
        ang_lay = QGridLayout(ang_grp)
        self.hud_p = self._add_hud_row(ang_lay, 0, "Rate p:")
        self.hud_q = self._add_hud_row(ang_lay, 1, "Rate q:")
        self.hud_r = self._add_hud_row(ang_lay, 2, "Rate r:")
        dash_layout.addWidget(ang_grp)

        dash_layout.addStretch()
        body_layout.addWidget(dash_panel)

        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        
        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("<b>Active View (Tab):</b>"))
        
        self.radio_group = QButtonGroup(self)
        self.radios: List[QRadioButton] = []
        
        # 1. ORDEM E NOMES ALTERADOS AQUI
        plot_options = ["3D Simulation", "Position", "Attitude", "Linear Velocity", "Angular Rate"]
        
        for i, text in enumerate(plot_options):
            rb = QRadioButton(text)
            if i == 0: rb.setChecked(True)
            self.radio_group.addButton(rb, i)
            view_controls.addWidget(rb)
            self.radios.append(rb)
            
        view_controls.addStretch()
        
        self.chk_autoscroll = QCheckBox(" Auto-Scroll (Live Window)")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.toggled.connect(self.init_graphs)
        view_controls.addWidget(self.chk_autoscroll)
        
        btn_reset_view = QPushButton(" Reset Zoom")
        btn_reset_view.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        btn_reset_view.clicked.connect(self.init_graphs)
        view_controls.addWidget(btn_reset_view)
        
        graph_layout.addLayout(view_controls)
        self.radio_group.idClicked.connect(lambda: self.init_graphs())
        
        self.right_stack = QStackedWidget()
        
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setStyleSheet("border-radius: 8px; border: 1px solid #444;")
        
        import time
        cache_buster = int(time.time())
        self.unity_widget = QWebEngineView()
        self.unity_widget.load(QUrl(f"http://127.0.0.1:{self.webgl_port}?reload={cache_buster}"))
        
        self.right_stack.addWidget(self.graph_widget)
        self.right_stack.addWidget(self.unity_widget)
        
        graph_layout.addWidget(self.right_stack, stretch=1)
        
        body_layout.addWidget(graph_panel, stretch=1)
        main_layout.addLayout(body_layout, stretch=1)

    def _create_hud_value(self, default_val: str, color: str = "#00d4ff", size: int = 16) -> QLabel:
        lbl = QLabel(default_val)
        font = QFont("Consolas", size, QFont.Weight.Bold)
        lbl.setFont(font)
        lbl.setStyleSheet(f"color: {color};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def _add_hud_row(self, layout: QGridLayout, row: int, title: str) -> QLabel:
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        lbl_val = self._create_hud_value("0.00")
        layout.addWidget(lbl_title, row, 0)
        layout.addWidget(lbl_val, row, 1)
        return lbl_val

    def receive_online_data(self, data: Dict[str, float]) -> None:
        if self.online_start_t is None:
            self.online_start_t = data['t']
            self.lbl_live_status.setText("● TELEMETRY LIVE")
            self.lbl_live_status.setStyleSheet("color: #ff3333; font-weight: bold; font-size: 14px;")
            
        rel_time = data['t'] - self.online_start_t
        self.lbl_time.setText(f"{rel_time:.2f} s")
        
        self.hud_n.setText(f"{data['x']:.2f}")
        self.hud_e.setText(f"{data['y']:.2f}")
        self.hud_d.setText(f"{-data['z']:.2f}")
        self.hud_roll.setText(f"{data['roll'] * 57.2958:.1f}")
        self.hud_pitch.setText(f"{data['pitch'] * 57.2958:.1f}")
        self.hud_yaw.setText(f"{data['yaw'] * 57.2958:.1f}")
        self.hud_u.setText(f"{data['u']:.2f}")
        self.hud_v.setText(f"{data['v']:.2f}")
        self.hud_w.setText(f"{data['w']:.2f}")
        self.hud_p.setText(f"{data['p']:.2f}")
        self.hud_q.setText(f"{data['q']:.2f}")
        self.hud_r.setText(f"{data['r']:.2f}")

        self.flight_history['t'].append(rel_time)
        self.flight_history['x'].append(data['x'])
        self.flight_history['y'].append(data['y'])
        self.flight_history['z'].append(-data['z']) 
        self.flight_history['roll'].append(data['roll'] * 57.2958)
        self.flight_history['pitch'].append(data['pitch'] * 57.2958)
        self.flight_history['yaw'].append(data['yaw'] * 57.2958)
        self.flight_history['u'].append(data['u'])
        self.flight_history['v'].append(data['v'])
        self.flight_history['w'].append(data['w'])
        self.flight_history['p'].append(data['p'])
        self.flight_history['q'].append(data['q'])
        self.flight_history['r'].append(data['r'])

        for key in self.online_data.keys():
            self.online_data[key] = np.roll(self.online_data[key], -1)
            
        self.online_data['t'][-1] = rel_time
        self.online_data['x'][-1] = data['x']
        self.online_data['y'][-1] = data['y']
        self.online_data['z'][-1] = -data['z'] 
        self.online_data['roll'][-1] = data['roll'] * 57.2958
        self.online_data['pitch'][-1] = data['pitch'] * 57.2958
        self.online_data['yaw'][-1] = data['yaw'] * 57.2958
        self.online_data['u'][-1] = data['u']
        self.online_data['v'][-1] = data['v']
        self.online_data['w'][-1] = data['w']
        self.online_data['p'][-1] = data['p']
        self.online_data['q'][-1] = data['q']
        self.online_data['r'][-1] = data['r']
        
        if self.graph_items:
            is_auto_scroll = self.chk_autoscroll.isChecked()
            for curve, _, data_key, _ in self.graph_items:
                if is_auto_scroll:
                    curve.setData(self.online_data['t'], self.online_data[data_key])
                else:
                    curve.setData(self.flight_history['t'], self.flight_history[data_key])

    def init_graphs(self) -> None:
        self.graph_widget.clear() 
        self.graph_items = []
        view_idx = self.radio_group.checkedId()
        
        # 2. ÍNDICES ATUALIZADOS AQUI NA LÓGICA
        if view_idx == 0: # Agora o índice 0 é a simulação 3D
            self.chk_autoscroll.setEnabled(False)
            self.right_stack.setCurrentIndex(1) # Traz o WebEngine para frente
            return

        self.right_stack.setCurrentIndex(0) # Traz os gráficos 2D para frente
        self.chk_autoscroll.setEnabled(True)
        
        pens = [pg.mkPen(color='#f44336', width=2), pg.mkPen(color='#4caf50', width=2), pg.mkPen(color='#00d4ff', width=2)] 

        def _create_plot(row: int, col: int, title: str, y_label: str) -> pg.PlotItem:
            p = self.graph_widget.addPlot(row=row, col=col)
            p.setLabel('left', y_label); p.setLabel('bottom', 'Time [s]')
            p.setTitle(title, color='#dddddd'); p.setMouseEnabled(x=True, y=True)
            p.enableAutoRange(axis='y'); p.setDownsampling(ds=True, auto=True, mode='peak'); p.setClipToView(True)
            if self.chk_autoscroll.isChecked(): p.enableAutoRange(axis='x') 
            p.showGrid(x=True, y=True, alpha=0.3)
            return p

        if view_idx == 1: # Índice 1 agora é Position
            keys = ['x', 'y', 'z']; labels = ['North [m]', 'East [m]', 'Altitude [m]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        elif view_idx == 2: # Índice 2 agora é Attitude
            keys = ['roll', 'pitch', 'yaw']; labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))
                
        elif view_idx == 3: # Índice 3 agora é Linear Velocity
            keys = ['u', 'v', 'w']; labels = ['u [m/s]', 'v [m/s]', 'w [m/s]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        elif view_idx == 4: # Índice 4 agora é Angular Rate
            keys = ['p', 'q', 'r']; labels = ['p [rad/s]', 'q [rad/s]', 'r [rad/s]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        if not self.chk_autoscroll.isChecked() and self.flight_history['t']:
            for curve, _, data_key, _ in self.graph_items:
                curve.setData(self.flight_history['t'], self.flight_history[data_key])

    def clear_history(self) -> None:
        msg = CustomMessageBox("Clear Data", "Are you sure you want to clear the telemetry history?", "You won't be able to export this flight anymore.", msg_type="question", parent=self)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.online_start_t = None
            self.flight_history = self._create_empty_history()
            self._init_rolling_buffer()
            self.lbl_live_status.setText("● SYSTEM STANDBY")
            self.lbl_live_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 14px;")
            self.init_graphs()

    def export_csv(self) -> None:
        if not self.flight_history['t']:
            msg = CustomMessageBox("Info", "No data to export.", "The flight history is currently empty.", msg_type="info", parent=self)
            msg.exec()
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Flight Log (CSV)", "", "CSV Files (*.csv)")
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                data_matrix = np.column_stack((
                    self.flight_history['t'], self.flight_history['x'], self.flight_history['y'], self.flight_history['z'],
                    self.flight_history['roll'], self.flight_history['pitch'], self.flight_history['yaw'],
                    self.flight_history['u'], self.flight_history['v'], self.flight_history['w'],
                    self.flight_history['p'], self.flight_history['q'], self.flight_history['r']
                ))
                header = ["Time_s", "Pos_N_m", "Pos_E_m", "Pos_Alt_m", "Roll_deg", "Pitch_deg", "Yaw_deg", "Vel_u_ms", "Vel_v_ms", "Vel_w_ms", "Rate_p_rads", "Rate_q_rads", "Rate_r_rads"]
                np.savetxt(file_path, data_matrix, delimiter=",", header=",".join(header), comments="", fmt='%.6f')
                msg = CustomMessageBox("Success", "Flight Log Exported!", f"Data saved to:\n{file_path}", msg_type="success", parent=self)
                msg.exec()
            except Exception as err:
                msg = CustomMessageBox("Export Error", "Failed to export CSV.", str(err), msg_type="error", parent=self)
                msg.exec()

    def export_graph_image(self) -> None:
        if not self.flight_history['t']:
            msg = CustomMessageBox("Info", "No data to export.", "The flight history is empty.", msg_type="info", parent=self)
            msg.exec()
            return

        dialog = ExportGraphDialog(self)
        if dialog.exec():
            idx = dialog.selected_index
            _, keys = dialog.options[idx]
            
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph Snapshot", "", "PNG Images (*.png)")
            if file_path:
                if not file_path.endswith('.png'): file_path += '.png'
                try:
                    if not keys: 
                        pixmap = self.graph_widget.grab()
                    else:
                        off_widget = pg.GraphicsLayoutWidget()
                        n_plots = len(keys)
                        cols = 3 if n_plots >= 3 else n_plots
                        rows = (n_plots + cols - 1) // cols
                        off_widget.resize(1100, 300 * rows)
                        off_widget.setBackground('#2b2b2b')
                        
                        labels_dict = {'x': 'North [m]', 'y': 'East [m]', 'z': 'Altitude [m]', 'roll': 'Roll [deg]', 'pitch': 'Pitch [deg]', 'yaw': 'Yaw [deg]', 'u': 'u [m/s]', 'v': 'v [m/s]', 'w': 'w [m/s]', 'p': 'p [rad/s]', 'q': 'q [rad/s]', 'r': 'r [rad/s]'}
                        pens = ['#f44336', '#4caf50', '#00d4ff']
                        
                        for i, key in enumerate(keys):
                            r_idx, c_idx = i // cols, i % cols
                            p = off_widget.addPlot(row=r_idx, col=c_idx)
                            p.setLabel('left', labels_dict[key]); p.setLabel('bottom', 'Time [s]')
                            p.showGrid(x=True, y=True, alpha=0.3)
                            curve = p.plot(pen=pg.mkPen(color=pens[c_idx], width=2))
                            curve.setData(self.flight_history['t'], self.flight_history[key])
                            
                        QApplication.processEvents() 
                        pixmap = off_widget.grab()
                        
                    pixmap.save(file_path, "PNG")
                    msg = CustomMessageBox("Success", "Snapshot saved!", f"High-resolution image rendered to:\n{file_path}", msg_type="success", parent=self)
                    msg.exec()
                except Exception as err:
                    msg = CustomMessageBox("Export Error", "Failed to render and save image.", str(err), msg_type="error", parent=self)
                    msg.exec()
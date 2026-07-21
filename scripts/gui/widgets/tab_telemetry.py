import sys
import numpy as np
import time
from pathlib import Path
import pyqtgraph as pg

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QMessageBox, QFileDialog, QGroupBox,
                               QRadioButton, QFrame, QButtonGroup, QDialog, 
                               QTextBrowser, QGridLayout, QCheckBox, QComboBox, QApplication,
                               QStyle)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QShortcut, QKeySequence, QFont, QPixmap

# =========================================================================
# CAIXAS DE DIÁLOGO
# =========================================================================
class TelemetryHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documentation - Live Telemetry")
        self.resize(800, 500) 
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        
        html_content = """
        <html>
        <head>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #e0e0e0; padding: 15px 25px; }
            h2 { color: #00d4ff; margin-top: 10px; border-bottom: 1px solid #555; padding-bottom: 5px;}
            h3 { color: #4caf50; margin-top: 25px; margin-bottom: 5px; }
            ul { margin-top: 5px; padding-left: 25px; }
            li { margin-bottom: 8px; }
            b { color: #ffffff; }
        </style>
        </head>
        <body>
        <h2>Live Telemetry Dashboard Guide</h2>
        <p>This panel acts as the Ground Control Station (GCS) for your UAV, monitoring data streamed directly from the ROS 2 environment in real-time.</p>
        
        <h3>1. Interactive Graphs & History</h3>
        <ul>
            <li><b>Pan & Zoom:</b> The graphs are fully interactive. Use the mouse wheel to zoom in/out, and click-drag to pan across the timeline.</li>
            <li><b>Auto-Scroll Toggle:</b> Uncheck the 'Auto-Scroll' box to stop the live window and view the complete flight history on the graphs.</li>
            <li><b>Switching Views:</b> Use the Radio Buttons or press <b>Tab</b> on your keyboard to quickly cycle between data categories, including the future 3D Digital Twin environment.</li>
        </ul>

        <h3>2. Data Recording & Export</h3>
        <ul>
            <li><b>Export CSV:</b> Generates a complete mathematical report of the current flight, saving the full history of all states for offline analysis in Matlab/Python.</li>
            <li><b>Snapshot Graph:</b> Advanced rendering tool. You can select specific data groups (e.g., Only Velocities, Only Altitude) and the system will render a high-resolution PNG off-screen specifically for your reports.</li>
            <li><b>Clear History:</b> Wipes the background memory and resets the dashboard timeline back to zero. Perfect to be used before triggering a new route.</li>
        </ul>
        </body>
        </html>
        """
        browser.setHtml(html_content)
        layout.addWidget(browser)

class CustomMessageBox(QDialog):
    def __init__(self, title, main_text, detail_text="", msg_type="info", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setStyleSheet("QDialog { background-color: #2b2b2b; } QLabel { color: #e0e0e0; font-size: 14px;} QPushButton { background-color: #0d6efd; color: white; border-radius: 4px; padding: 6px 16px; font-weight: bold; min-width: 80px; } QPushButton:hover { background-color: #0b5ed7; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel(f"<h3 style='margin: 0;'>{main_text}</h3>"))
        if detail_text:
            layout.addWidget(QLabel(detail_text))
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

class ExportGraphDialog(QDialog):
    """Menu avançado para escolher e renderizar qual gráfico salvar, conforme solicitado."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Export Options")
        self.setMinimumWidth(400)
        self.setStyleSheet("QDialog { background-color: #2b2b2b; color: white; } QLabel { color: #e0e0e0; font-size: 14px;} QPushButton { background-color: #0d6efd; color: white; border-radius: 4px; padding: 6px 16px; font-weight: bold; } QComboBox { background-color: #3b3b3b; color: white; padding: 5px; border: 1px solid #555; font-size: 13px;}")
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the specific data group to render and export:"))
        
        self.combo = QComboBox()
        self.options = [
            ("Current Visible Screen (WYSIWYG)", []),
            ("Full Dashboard Overview (12 Plots)", ['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']),
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

    def accept(self):
        self.selected_index = self.combo.currentIndex()
        super().accept()

# Configuração global do PyQtGraph para alta performance
pg.setConfigOption('background', '#2b2b2b')
pg.setConfigOption('foreground', '#aaaaaa')
pg.setConfigOptions(antialias=True) 

# =========================================================================
# CLASSE PRINCIPAL DA ABA
# =========================================================================
class TabTelemetry(QWidget):
    def __init__(self, main_window_ref):
        super().__init__()
        self.main_window = main_window_ref 
        
        self.window_size = 300 
        self.online_start_t = None
        self._init_rolling_buffer()
        self.flight_history = self._create_empty_history()
        self.graph_items = [] 
        
        self._setup_dark_theme()
        self._build_ui()
        self._setup_shortcuts()
        self.init_graphs()

    def _init_rolling_buffer(self):
        t_array = np.linspace(-10.0, 0.0, self.window_size)
        self.online_data = {
            't': t_array.copy(),
            'x': np.zeros(self.window_size), 'y': np.zeros(self.window_size), 'z': np.zeros(self.window_size),
            'roll': np.zeros(self.window_size), 'pitch': np.zeros(self.window_size), 'yaw': np.zeros(self.window_size),
            'u': np.zeros(self.window_size), 'v': np.zeros(self.window_size), 'w': np.zeros(self.window_size),
            'p': np.zeros(self.window_size), 'q': np.zeros(self.window_size), 'r': np.zeros(self.window_size)
        }

    def _create_empty_history(self):
        return {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 
                'u': [], 'v': [], 'w': [], 'p': [], 'q': [], 'r': []}

    def _setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        self.setPalette(palette)
        self.setStyleSheet("""
            QGroupBox { border: 1px solid #555555; border-radius: 4px; margin-top: 2ex; padding-top: 10px; color: #e0e0e0; font-weight: bold;}
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #00d4ff; }
            QPushButton { background-color: #3b3b3b; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #0d6efd; border: 1px solid #0d6efd; }
            QCheckBox { color: white; font-weight: bold; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; background: #2b2b2b; }
            QCheckBox::indicator:checked { background: #00d4ff; border: 1px solid #00d4ff; }
        """)

    def _setup_shortcuts(self):
        self.shortcut_cycle = QShortcut(QKeySequence("Tab"), self)
        self.shortcut_cycle.activated.connect(self.cycle_views)
        self.shortcut_help = QShortcut(QKeySequence("F1"), self)
        self.shortcut_help.activated.connect(self.show_help_window)

    def cycle_views(self):
        current_id = self.radio_group.checkedId()
        next_id = (current_id + 1) % len(self.radios)
        self.radios[next_id].setChecked(True)
        self.init_graphs()

    def show_help_window(self):
        self.help_dialog = TelemetryHelpDialog(self)
        self.help_dialog.show()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # --- TOP TOOLBAR ---
        toolbar_layout = QHBoxLayout()
        self.lbl_live_status = QLabel("● SYSTEM STANDBY")
        self.lbl_live_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 14px;")
        toolbar_layout.addWidget(self.lbl_live_status)
        toolbar_layout.addStretch()
        
        self.btn_clear = QPushButton(" Clear History"); self.btn_clear.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)); self.btn_clear.clicked.connect(self.clear_history)
        self.btn_export_img = QPushButton(" Snapshot Graph"); self.btn_export_img.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)); self.btn_export_img.clicked.connect(self.export_graph_image)
        self.btn_export_csv = QPushButton(" Export Flight CSV"); self.btn_export_csv.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon)); self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_help = QPushButton(" Help"); self.btn_help.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)); self.btn_help.clicked.connect(self.show_help_window)
        
        toolbar_layout.addWidget(self.btn_clear); toolbar_layout.addWidget(self.btn_export_img); toolbar_layout.addWidget(self.btn_export_csv); toolbar_layout.addWidget(self.btn_help)
        main_layout.addLayout(toolbar_layout)

        # --- BODY SPLIT ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        
        dash_panel = QFrame(); dash_panel.setFixedWidth(280)
        dash_layout = QVBoxLayout(dash_panel); dash_layout.setContentsMargins(0, 0, 0, 0); dash_layout.setSpacing(10)

        time_grp = QGroupBox("Mission Time")
        time_lay = QVBoxLayout(time_grp)
        self.lbl_time = self._create_hud_value("0.00 s", color="#00ff00", size=24)
        time_lay.addWidget(self.lbl_time)
        dash_layout.addWidget(time_grp)

        pos_grp = QGroupBox("Global Position [m]")
        pos_lay = QGridLayout(pos_grp)
        self.hud_n = self._add_hud_row(pos_lay, 0, "North (X):"); self.hud_e = self._add_hud_row(pos_lay, 1, "East (Y):"); self.hud_d = self._add_hud_row(pos_lay, 2, "Altitude (-Z):")
        dash_layout.addWidget(pos_grp)

        att_grp = QGroupBox("Attitude [deg]")
        att_lay = QGridLayout(att_grp)
        self.hud_roll = self._add_hud_row(att_lay, 0, "Roll:"); self.hud_pitch = self._add_hud_row(att_lay, 1, "Pitch:"); self.hud_yaw = self._add_hud_row(att_lay, 2, "Yaw:")
        dash_layout.addWidget(att_grp)

        lin_grp = QGroupBox("Linear Velocity [m/s]")
        lin_lay = QGridLayout(lin_grp)
        self.hud_u = self._add_hud_row(lin_lay, 0, "Vel u:"); self.hud_v = self._add_hud_row(lin_lay, 1, "Vel v:"); self.hud_w = self._add_hud_row(lin_lay, 2, "Vel w:")
        dash_layout.addWidget(lin_grp)

        ang_grp = QGroupBox("Angular Rate [rad/s]")
        ang_lay = QGridLayout(ang_grp)
        self.hud_p = self._add_hud_row(ang_lay, 0, "Rate p:"); self.hud_q = self._add_hud_row(ang_lay, 1, "Rate q:"); self.hud_r = self._add_hud_row(ang_lay, 2, "Rate r:")
        dash_layout.addWidget(ang_grp)

        dash_layout.addStretch()
        body_layout.addWidget(dash_panel)

        # 2. PAINEL DIREITO: GRÁFICOS INTERATIVOS E UNITY
        graph_panel = QWidget(); graph_layout = QVBoxLayout(graph_panel); graph_layout.setContentsMargins(0, 0, 0, 0)
        
        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("<b>Active View (Tab):</b>"))
        
        self.radio_group = QButtonGroup(self)
        self.radios = []
        plot_options = ["Position", "Attitude", "Linear Velocity", "Angular Rate", "3D Digital Twin (Unity)"]
        
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
        
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setStyleSheet("border-radius: 8px; border: 1px solid #444;")
        graph_layout.addWidget(self.graph_widget, stretch=1)
        
        body_layout.addWidget(graph_panel, stretch=1)
        main_layout.addLayout(body_layout, stretch=1)

    def _create_hud_value(self, default_val, color="#00d4ff", size=16):
        lbl = QLabel(default_val); font = QFont("Consolas", size, QFont.Weight.Bold); lbl.setFont(font)
        lbl.setStyleSheet(f"color: {color};"); lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def _add_hud_row(self, layout, row, title):
        lbl_title = QLabel(title); lbl_title.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        lbl_val = self._create_hud_value("0.00")
        layout.addWidget(lbl_title, row, 0); layout.addWidget(lbl_val, row, 1)
        return lbl_val

    # =========================================================================
    # RECEPTOR DO ROS 2 E ATUALIZAÇÃO DA TELA
    # =========================================================================
    def receive_online_data(self, data: dict):
        if self.online_start_t is None:
            self.online_start_t = data['t']
            self.lbl_live_status.setText("● TELEMETRY LIVE")
            self.lbl_live_status.setStyleSheet("color: #ff3333; font-weight: bold; font-size: 14px;")
            
        rel_time = data['t'] - self.online_start_t

        self.lbl_time.setText(f"{rel_time:.2f} s")
        self.hud_n.setText(f"{data['x']:.2f}"); self.hud_e.setText(f"{data['y']:.2f}"); self.hud_d.setText(f"{-data['z']:.2f}")
        self.hud_roll.setText(f"{data['roll'] * 57.2958:.1f}"); self.hud_pitch.setText(f"{data['pitch'] * 57.2958:.1f}"); self.hud_yaw.setText(f"{data['yaw'] * 57.2958:.1f}")
        self.hud_u.setText(f"{data['u']:.2f}"); self.hud_v.setText(f"{data['v']:.2f}"); self.hud_w.setText(f"{data['w']:.2f}")
        self.hud_p.setText(f"{data['p']:.2f}"); self.hud_q.setText(f"{data['q']:.2f}"); self.hud_r.setText(f"{data['r']:.2f}")

        self.flight_history['t'].append(rel_time)
        self.flight_history['x'].append(data['x']); self.flight_history['y'].append(data['y']); self.flight_history['z'].append(-data['z']) 
        self.flight_history['roll'].append(data['roll'] * 57.2958); self.flight_history['pitch'].append(data['pitch'] * 57.2958); self.flight_history['yaw'].append(data['yaw'] * 57.2958)
        self.flight_history['u'].append(data['u']); self.flight_history['v'].append(data['v']); self.flight_history['w'].append(data['w'])
        self.flight_history['p'].append(data['p']); self.flight_history['q'].append(data['q']); self.flight_history['r'].append(data['r'])

        for key in self.online_data.keys():
            self.online_data[key] = np.roll(self.online_data[key], -1)
            
        self.online_data['t'][-1] = rel_time
        self.online_data['x'][-1] = data['x']; self.online_data['y'][-1] = data['y']; self.online_data['z'][-1] = -data['z'] 
        self.online_data['roll'][-1] = data['roll'] * 57.2958; self.online_data['pitch'][-1] = data['pitch'] * 57.2958; self.online_data['yaw'][-1] = data['yaw'] * 57.2958
        self.online_data['u'][-1] = data['u']; self.online_data['v'][-1] = data['v']; self.online_data['w'][-1] = data['w']
        self.online_data['p'][-1] = data['p']; self.online_data['q'][-1] = data['q']; self.online_data['r'][-1] = data['r']
        
        if self.graph_items:
            is_auto_scroll = self.chk_autoscroll.isChecked()
            for curve, plot_widget, data_key, _ in self.graph_items:
                if is_auto_scroll:
                    curve.setData(self.online_data['t'], self.online_data[data_key])
                else:
                    curve.setData(self.flight_history['t'], self.flight_history[data_key])

    # =========================================================================
    # INICIALIZAÇÃO DE GRÁFICOS (E UNITY PLACEHOLDER)
    # =========================================================================
    def init_graphs(self):
        self.graph_widget.clear() 
        self.graph_items = []
        view_idx = self.radio_group.checkedId()
        
        pens = [pg.mkPen(color='#f44336', width=2), pg.mkPen(color='#4caf50', width=2), pg.mkPen(color='#00d4ff', width=2)] 

        def _create_plot(row, col, title, y_label):
            p = self.graph_widget.addPlot(row=row, col=col)
            p.setLabel('left', y_label); p.setLabel('bottom', 'Time [s]')
            p.setTitle(title, color='#dddddd')
            p.setMouseEnabled(x=True, y=True); p.enableAutoRange(axis='y') 
            p.setDownsampling(ds=True, auto=True, mode='peak'); p.setClipToView(True)
            if self.chk_autoscroll.isChecked(): p.enableAutoRange(axis='x') 
            p.showGrid(x=True, y=True, alpha=0.3)
            return p

        if view_idx == 0: 
            keys = ['x', 'y', 'z']; labels = ['North [m]', 'East [m]', 'Altitude [m]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        elif view_idx == 1: 
            keys = ['roll', 'pitch', 'yaw']; labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))
                
        elif view_idx == 2: 
            keys = ['u', 'v', 'w']; labels = ['u [m/s]', 'v [m/s]', 'w [m/s]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        elif view_idx == 3: 
            keys = ['p', 'q', 'r']; labels = ['p [rad/s]', 'q [rad/s]', 'r [rad/s]']
            for i in range(3):
                p = _create_plot(0, i, labels[i], labels[i]); curve = p.plot(pen=pens[i])
                self.graph_items.append((curve, p, keys[i], labels[i]))

        elif view_idx == 4:
            self.chk_autoscroll.setEnabled(False)
            p = self.graph_widget.addPlot(); p.hideAxis('left'); p.hideAxis('bottom')
            html_text = '<div style="text-align: center"><span style="color: #00d4ff; font-size: 24pt;">Unity 3D Digital Twin</span><br><br><span style="color: #aaaaaa; font-size: 14pt;">Placeholder ready for WebEngine integration</span></div>'
            text_item = pg.TextItem(html=html_text, anchor=(0.5, 0.5)); p.addItem(text_item); text_item.setPos(0.5, 0.5) 
            return

        self.chk_autoscroll.setEnabled(True)
        if not self.chk_autoscroll.isChecked() and self.flight_history['t']:
            for curve, _, data_key, _ in self.graph_items:
                curve.setData(self.flight_history['t'], self.flight_history[data_key])

    # =========================================================================
    # FERRAMENTAS EXTRAS (CLEAR, CSV E IMAGEM OFF-SCREEN)
    # =========================================================================
    def clear_history(self):
        msg = CustomMessageBox("Clear Data", "Are you sure you want to clear the telemetry history?", "You won't be able to export this flight anymore.", msg_type="question", parent=self)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.online_start_t = None
            self.flight_history = self._create_empty_history()
            self._init_rolling_buffer()
            self.lbl_live_status.setText("● SYSTEM STANDBY")
            self.lbl_live_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 14px;")
            self.init_graphs()

    def export_csv(self):
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
                header = ["Time_s", "Pos_N_m", "Pos_E_m", "Pos_Alt_m", "Roll_deg", "Pitch_deg", "Yaw_deg", 
                          "Vel_u_ms", "Vel_v_ms", "Vel_w_ms", "Rate_p_rads", "Rate_q_rads", "Rate_r_rads"]
                np.savetxt(file_path, data_matrix, delimiter=",", header=",".join(header), comments="", fmt='%.6f')
                msg = CustomMessageBox("Success", "Flight Log Exported!", f"Data saved to:\n{file_path}", msg_type="success", parent=self)
                msg.exec()
            except Exception as e:
                msg = CustomMessageBox("Export Error", "Failed to export CSV.", str(e), msg_type="error", parent=self)
                msg.exec()

    def export_graph_image(self):
        if not self.flight_history['t']:
            msg = CustomMessageBox("Info", "No data to export.", "The flight history is empty.", msg_type="info", parent=self)
            msg.exec()
            return

        dialog = ExportGraphDialog(self)
        if dialog.exec():
            idx = dialog.selected_index
            label_text, keys = dialog.options[idx]
            
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph Snapshot", "", "PNG Images (*.png)")
            if file_path:
                if not file_path.endswith('.png'): file_path += '.png'
                try:
                    if not keys: 
                        pixmap = self.graph_widget.grab()
                    else:
                        # RENDERIZAÇÃO INVISÍVEL OFF-SCREEN
                        off_widget = pg.GraphicsLayoutWidget()
                        
                        # Calcula quantas linhas o grid terá (max 3 colunas por linha)
                        n_plots = len(keys)
                        cols = 3 if n_plots >= 3 else n_plots
                        rows = (n_plots + cols - 1) // cols
                        
                        # Tamanho dinâmico para não achatar
                        off_widget.resize(1100, 300 * rows)
                        off_widget.setBackground('#2b2b2b')
                        
                        labels_dict = {'x': 'North [m]', 'y': 'East [m]', 'z': 'Altitude [m]',
                                       'roll': 'Roll [deg]', 'pitch': 'Pitch [deg]', 'yaw': 'Yaw [deg]',
                                       'u': 'u [m/s]', 'v': 'v [m/s]', 'w': 'w [m/s]',
                                       'p': 'p [rad/s]', 'q': 'q [rad/s]', 'r': 'r [rad/s]'}
                        
                        pens = ['#f44336', '#4caf50', '#00d4ff']
                        
                        for i, key in enumerate(keys):
                            r = i // cols
                            c = i % cols
                            p = off_widget.addPlot(row=r, col=c)
                            p.setLabel('left', labels_dict[key])
                            p.setLabel('bottom', 'Time [s]')
                            p.showGrid(x=True, y=True, alpha=0.3)
                            
                            curve = p.plot(pen=pg.mkPen(color=pens[c], width=2))
                            curve.setData(self.flight_history['t'], self.flight_history[key])
                            
                        QApplication.processEvents() 
                        pixmap = off_widget.grab()
                        
                    pixmap.save(file_path, "PNG")
                    msg = CustomMessageBox("Success", "Snapshot saved!", f"High-resolution image rendered to:\n{file_path}", msg_type="success", parent=self)
                    msg.exec()
                except Exception as e:
                    msg = CustomMessageBox("Export Error", "Failed to render and save image.", str(e), msg_type="error", parent=self)
                    msg.exec()
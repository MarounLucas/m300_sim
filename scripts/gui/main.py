"""Launch the PySide6 Ground Control Station GUI and isolated ROS 2 bridge process."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys

# Inject source and installed scripts roots into Python path for local imports.
_src_scripts = '/workspace/src/m300_sim/scripts'
_src_root = '/workspace/src/m300_sim'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in (project_root, _src_scripts, _src_root):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# Configure QtWebEngine Chromium flags before importing PySide6/QtWebEngine so
# WebGL and Docker shared-memory settings (--disable-dev-shm-usage) take effect.
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = (
    '--no-sandbox --disable-dev-shm-usage --ignore-gpu-blocklist '
    '--enable-webgl --enable-gpu-rasterization'
)
os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '9999'

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.ros_bridge import run_bridge  # noqa: E402
from gui.widgets.main_window import MainWindow  # noqa: E402
import paths  # noqa: E402


def apply_global_dark_theme(app: QApplication) -> None:
    """Apply a dark Fusion palette across the Qt application."""
    app.setStyle('Fusion')
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.Base, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    app.setPalette(palette)


def main(args: list[str] | None = None) -> None:
    """Start the isolated ROS 2 bridge process and run the Qt GUI event loop."""
    paths.sync_ros_install_scripts()

    # Spawn the ROS 2 bridge in a separate process so rclpy/FastDDS and
    # PySide6/QtWebEngine never share a C++ heap in the same process.
    ctx = mp.get_context('spawn')
    to_gui: mp.Queue = ctx.Queue(maxsize=64)
    from_gui: mp.Queue = ctx.Queue(maxsize=32)
    ros_args = list(sys.argv if args is None else args)

    bridge_proc = ctx.Process(
        target=run_bridge,
        args=(to_gui, from_gui, ros_args),
        daemon=True,
    )
    bridge_proc.start()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    apply_global_dark_theme(app)

    window = MainWindow()

    def _send_start() -> None:
        try:
            from_gui.put_nowait(('start', None))
        except queue.Full:
            pass

    def _send_nn(enabled: bool) -> None:
        state = bool(enabled)
        try:
            from_gui.put_nowait(('nn', state))
        except queue.Full:
            pass
        window.tab_inputs.set_nn_button_state(state)
        window.tab_outputs.set_nn_button_state(state)

    window.tab_inputs.btn_run_sim.clicked.connect(_send_start)
    window.tab_inputs.nn_toggled.connect(_send_nn)
    window.tab_outputs.nn_toggled.connect(_send_nn)

    def _poll_bridge_queue() -> None:
        latest_telemetry: dict[str, float] | None = None
        while True:
            try:
                kind, payload = to_gui.get_nowait()
            except queue.Empty:
                break
            if kind == 'telemetry' and isinstance(payload, dict):
                latest_telemetry = payload
            elif kind == 'nn':
                state = bool(payload)
                window.tab_inputs.set_nn_button_state(state)
                window.tab_outputs.set_nn_button_state(state)
        if latest_telemetry is not None:
            window.tab_outputs.receive_online_data(latest_telemetry)

    poll_timer = QTimer()
    poll_timer.timeout.connect(_poll_bridge_queue)
    poll_timer.start(25)

    window.show()
    exit_code = app.exec()

    poll_timer.stop()
    try:
        from_gui.put_nowait(('quit', None))
    except Exception:
        pass
    bridge_proc.join(timeout=1.5)
    if bridge_proc.is_alive():
        bridge_proc.terminate()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

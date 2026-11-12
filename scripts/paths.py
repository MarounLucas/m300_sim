"""Directory path resolution and initialization for the M350 ROS 2 simulator.

Defines canonical paths for configuration files, aircraft models, offline
missions, flight logs, and neural network artifacts relative to the package
root (located via ``package.xml``), and keeps installed ROS 2 Python scripts
synchronized with the source tree when running inside a container build.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` atomically via a temporary sibling file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'.{dst.name}.tmp.{os.getpid()}')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def get_project_root() -> Path:
    """Locate the package root directory by searching upwards for ``package.xml``.

    Returns:
        Absolute path to the ``m300_sim`` package root directory.
    """
    current_path = Path(__file__).resolve().parent

    for _ in range(6):
        if (current_path / 'package.xml').exists():
            return current_path
        current_path = current_path.parent

    return Path('/workspace/src/m300_sim')


PROJECT_ROOT: Path = get_project_root()
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent.parent
DATA_DIR: Path = PROJECT_ROOT / 'data'

AIRCRAFT_MODELS_DIR: Path = DATA_DIR / 'aircraft_models'
MISSIONS_DIR: Path = DATA_DIR / 'offline_missions'
ROS_CONFIG_DIR: Path = PROJECT_ROOT / 'config'
FLIGHT_GRAPHS_DIR: Path = DATA_DIR / 'flight_graphs'
FLIGHT_LOGS_DIR: Path = DATA_DIR / 'flight_logs'
NN_MODELS_DIR: Path = DATA_DIR / 'nn_models'
NN_VALIDATION_DIR: Path = DATA_DIR / 'nn_validation'

ROS_INSTALL_ROOT: Path = WORKSPACE_ROOT / 'install' / 'm300_sim'
ROS_INSTALL_DIR: Path = ROS_INSTALL_ROOT / 'share' / 'm300_sim' / 'config'


def sync_ros_install_scripts() -> int:
    """Synchronize modified source Python modules into the ROS 2 install tree.

    When ``colcon build`` is run without ``--symlink-install`` inside the
    container, ``install/m300_sim/lib/python*/site-packages/scripts`` holds a
    static copy of ``scripts/``. Copying updated ``.py`` files ensures live
    nodes and restarts always execute the latest source code.

    Returns:
        Number of Python files updated in the install tree.
    """
    lib_dir = ROS_INSTALL_ROOT / 'lib'
    src_scripts = PROJECT_ROOT / 'scripts'
    if not lib_dir.exists() or not src_scripts.exists():
        return 0

    updated = 0
    for site_scripts in lib_dir.glob('python*/site-packages/scripts'):
        if not site_scripts.is_dir() or site_scripts.is_symlink():
            continue
        for src_file in src_scripts.rglob('*.py'):
            rel_path = src_file.relative_to(src_scripts)
            dst_file = site_scripts / rel_path
            try:
                if dst_file.is_symlink():
                    continue
                need_copy = not dst_file.exists()
                if not need_copy:
                    src_stat = src_file.stat()
                    dst_stat = dst_file.stat()
                    need_copy = (
                        src_stat.st_size != dst_stat.st_size
                        or src_stat.st_mtime > dst_stat.st_mtime + 0.01
                    )
                if need_copy:
                    _atomic_copy(src_file, dst_file)
                    updated += 1
            except OSError:
                continue
        for legacy_dir_name in ('manual_sim', 'flight_log_process'):
            legacy_dir = site_scripts / legacy_dir_name
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir, ignore_errors=True)
        for legacy_dyn_file in ('nn_features.py', 'nn_residual.py'):
            legacy_f = site_scripts / 'dynamic' / legacy_dyn_file
            if legacy_f.exists():
                try:
                    legacy_f.unlink()
                except OSError:
                    pass

    for ep_file in lib_dir.glob('python*/site-packages/m300_sim*.egg-info/entry_points.txt'):
        try:
            txt = ep_file.read_text(encoding='utf-8')
            if 'scripts.manual_sim.joystick_node' in txt:
                ep_file.write_text(
                    txt.replace(
                        'scripts.manual_sim.joystick_node',
                        'scripts.teleop.joystick_node',
                    ),
                    encoding='utf-8',
                )
        except OSError:
            continue

    joy_bin = lib_dir / 'm300_sim' / 'joy_sub'
    if joy_bin.is_file():
        try:
            txt = joy_bin.read_text(encoding='utf-8')
            if 'scripts.manual_sim.joystick_node' in txt:
                joy_bin.write_text(
                    txt.replace(
                        'scripts.manual_sim.joystick_node',
                        'scripts.teleop.joystick_node',
                    ),
                    encoding='utf-8',
                )
        except OSError:
            pass

    launch_src = PROJECT_ROOT / 'launch' / 'quad_simulation.launch.py'
    launch_dst = (
        ROS_INSTALL_ROOT / 'share' / 'm300_sim' / 'launch' / 'quad_simulation.launch.py'
    )
    if launch_src.is_file() and launch_dst.parent.is_dir():
        try:
            if (
                not launch_dst.exists()
                or launch_src.stat().st_size != launch_dst.stat().st_size
                or launch_src.stat().st_mtime > launch_dst.stat().st_mtime + 0.01
            ):
                _atomic_copy(launch_src, launch_dst)
        except OSError:
            pass

    if updated > 0 and sys.argv and sys.argv[0].endswith('/gui_node'):
        if os.environ.get('_M300_GUI_REEXEC') != '1':
            os.environ['_M300_GUI_REEXEC'] = '1'
            os.execv(sys.executable, [sys.executable, *sys.argv])

    return updated


def initialize_directories() -> None:
    """Create the data and configuration directories required by the simulator."""
    AIRCRAFT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ROS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    FLIGHT_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    FLIGHT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    NN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NN_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if not ROS_INSTALL_DIR.exists():
        try:
            ROS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    sync_ros_install_scripts()


initialize_directories()

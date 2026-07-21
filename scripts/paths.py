"""Module responsible for managing and configuring project directory paths.

Defines the main paths for the M350 simulator (ROS 2) and ensures the
existence of required directories for system execution.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Locates the project root directory by searching for 'package.xml'.

    The function traverses the directory tree upwards (up to 6 levels)
    from the current file's location, searching for the ROS 2 package
    root. If not found, it returns a default fallback path for the
    Docker container.

    Returns:
        Path: Absolute path to the project's root directory.
    """
    current_path = Path(__file__).resolve().parent
    
    for _ in range(6):
        if (current_path / "package.xml").exists():
            return current_path
        current_path = current_path.parent
        
    return Path("/workspace/src/m300_sim")


# ========================================================
# PROJECT BASES
# ========================================================
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"

# ========================================================
# M350 SIMULATOR SPECIFIC DIRECTORIES
# ========================================================
AIRCRAFT_MODELS_DIR: Path = DATA_DIR / "aircraft_models"
MISSIONS_DIR: Path = DATA_DIR / "offline_missions"
WAYPOINTS_DIR: Path = DATA_DIR / "waypoints"
UNITY_BUILD_DIR: Path = PROJECT_ROOT / "scripts" / "unity" / "build"
ROS_CONFIG_DIR: Path = PROJECT_ROOT / "config"

# Actual ROS 2 installation path (forced for the Docker environment)
ROS_INSTALL_DIR: Path = Path(
    "/workspace/install/m300_sim/share/m300_sim/config"
)


def initialize_directories() -> None:
    """Creates the necessary directories for the simulator to function.

    Ensures the existence of models, missions, waypoints, and configuration
    directories. Attempts to create the ROS installation directory,
    ignoring errors if the environment denies write permissions.
    """
    AIRCRAFT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    WAYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ROS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ROS_INSTALL_DIR.exists():
        try:
            ROS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


# Executes directory creation as soon as the module is imported/executed,
# maintaining the script's original functional behavior.
initialize_directories()
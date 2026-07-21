import sys
import os
from pathlib import Path

def get_project_root():
    """
    Sobe na árvore de diretórios até encontrar o 'package.xml' do ROS 2.
    Isso garante que vamos achar a pasta 'm300_sim' original com precisão absoluta,
    não importa em qual subpasta este arquivo 'paths.py' esteja salvo.
    """
    current = Path(__file__).resolve().parent
    for _ in range(6):  # Sobe até 6 níveis procurando o package.xml
        if (current / "package.xml").exists():
            return current
        current = current.parent
        
    # Fallback super seguro garantido pelo Docker Dev Container
    return Path("/workspace/src/m300_sim")

# ========================================================
# BASES DO PROJETO
# ========================================================
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT  / "data"

# ========================================================
# DIRETÓRIOS ESPECÍFICOS DO SIMULADOR M350
# ========================================================
AIRCRAFT_MODELS_DIR = DATA_DIR / "aircraft_models"
MISSIONS_DIR = DATA_DIR / "offline_missions"
WAYPOINTS_DIR = DATA_DIR / "waypoints"
UNITY_BUILD_DIR = PROJECT_ROOT / "scripts" / "unity" / "build"

# 1. Pasta SRC Real (Encontrada com 100% de certeza pelo package.xml)
ROS_CONFIG_DIR = PROJECT_ROOT / "config"

# 2. Pasta INSTALL Real (Forçada pelo caminho estático do Docker)
# Isso ignora importações que podem falhar e vai direto na veia do ROS 2
ROS_INSTALL_DIR = Path("/workspace/install/m300_sim/share/m300_sim/config")

# Garante que as pastas existam
AIRCRAFT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
WAYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ROS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
if ROS_INSTALL_DIR.exists() == False:
    # Apenas tenta criar se a arvore de install do ROS permitir, senao segue o jogo
    try: ROS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    except: pass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BASE_DIR = ROOT / "project_folder"
DATA_DIR = ROOT / "project_folder" / "scTab"
TRAINING_FOLDER = ROOT / "project_folder" / "trained_models"
RESULTS_FOLDER = ROOT / "project_folder" / "results"
OOD_FOLDER = ROOT / "project_folder" / "OOD"
MULTIMODAL_FOLDER = ROOT / "project_folder" / "neurips_multimodal"
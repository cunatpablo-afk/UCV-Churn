from pathlib import Path


def project_root() -> Path:
    """Return the repository root from anywhere inside the project."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(path: Path) -> str:
    """Return a stable repo-relative path for reports and JSON metadata."""
    return path.resolve().relative_to(ROOT).as_posix()


ROOT = project_root()
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELING_DIR = ROOT / "data" / "modeling"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_REPORTS_DIR = REPORTS_DIR / "models"
MODELS_DIR = ROOT / "models"

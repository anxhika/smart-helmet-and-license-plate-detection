from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class YOLO11Config:
    classes: List[str] = field(default_factory=lambda: ["Helmet", "No Helmet", "Rider", "LP"])
    dataset_root: Path = PROJECT_ROOT / "dataset_new"
    dataset_yaml: Path = PROJECT_ROOT / "dataset_new" / "data.yaml"
    model_path: str = "yolo11n.pt"
    train_best_weights_path: Path = PROJECT_ROOT / "yolo11" / "runs" / "yolo11n_dataset_new" / "weights" / "best.pt"
    trained_model_path: Path = PROJECT_ROOT / "yolo11" / "weights" / "yolo11n_dataset_new.pt"
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16
    output_root: Path = PROJECT_ROOT / "yolo11" / "outputs"
    train_project_dir: Path = PROJECT_ROOT / "yolo11" / "runs"
    train_name: str = "yolo11n_dataset_new"
    weights_dir: Path = PROJECT_ROOT / "yolo11" / "weights"
    plate_recognizer_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
    plate_recognizer_region: str = "in"
    plate_recognizer_token_env: str = "PLATE_RECOGNIZER_TOKEN"
    plate_recognizer_timeout_sec: int = 15


def _normalize_names(names_obj) -> List[str]:
    if isinstance(names_obj, dict):
        return [str(names_obj[idx]) for idx in sorted(names_obj.keys(), key=lambda k: int(k))]
    if isinstance(names_obj, list):
        return [str(name) for name in names_obj]
    raise ValueError("Invalid 'names' format in dataset yaml. Expected list or dict.")


def load_dataset_class_names(dataset_yaml: Path) -> List[str]:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"'names' is missing in dataset yaml: {dataset_yaml}")
    return _normalize_names(data["names"])


def ensure_dataset_class_order(config: YOLO11Config) -> None:
    dataset_names = load_dataset_class_names(config.dataset_yaml)
    if list(config.classes) != dataset_names:
        raise ValueError(
            "Class order mismatch between config and dataset_new/data.yaml. "
            f"config={list(config.classes)} dataset={dataset_names}"
        )


def resolve_model_path(config: YOLO11Config, model_path: str | None = None) -> str:
    if model_path:
        candidate = Path(model_path)
        return str(candidate if candidate.exists() else model_path)
    if config.train_best_weights_path.exists():
        return str(config.train_best_weights_path)
    if config.trained_model_path.exists():
        return str(config.trained_model_path)
    return config.model_path


def ensure_output_dirs(config: YOLO11Config) -> None:
    (config.output_root / "images").mkdir(parents=True, exist_ok=True)
    (config.output_root / "videos").mkdir(parents=True, exist_ok=True)
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    config.train_project_dir.mkdir(parents=True, exist_ok=True)


def expected_classes(config: YOLO11Config) -> Sequence[str]:
    return tuple(config.classes)


def get_plate_recognizer_token(config: YOLO11Config) -> str | None:
    token = os.getenv(config.plate_recognizer_token_env, "").strip()
    return token if token else None


DEFAULT_CONFIG = YOLO11Config()

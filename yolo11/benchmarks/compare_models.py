import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from yolo11.config import DEFAULT_CONFIG, PROJECT_ROOT, resolve_model_path
from yolo11.evaluator import evaluate_detection_model, evaluate_yolov5_native


def _resolve_yolov5_weights(path: Optional[str]) -> str:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"YOLOv5 weights not found: {path}")

    default_v5 = PROJECT_ROOT / "runs" / "train" / "finalModel" / "weights" / "best.pt"
    if not default_v5.exists():
        raise FileNotFoundError(
            f"YOLOv5 weights not found at default path: {default_v5}. "
            "Pass --yolov5-weights explicitly."
        )
    return str(default_v5)


def _delete_split_cache(dataset_yaml: Path, split: str) -> None:
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    split_path = data_cfg.get(split) or data_cfg.get("test") or data_cfg.get("val")
    if not split_path:
        return
    root = data_cfg.get("path", "")
    split_path = Path(split_path)
    if not split_path.is_absolute():
        split_path = (Path(root) / split_path) if root else (dataset_yaml.parent / split_path)
    labels_guess = Path(str(split_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/"))
    cache_file = labels_guess.with_suffix(".cache")
    if cache_file.exists():
        cache_file.unlink()


def _build_rows(results: Dict[str, Dict]) -> List[Dict[str, str]]:
    rows = []
    for model_name, result in results.items():
        m = result["metrics"]
        rows.append(
            {
                "Model": model_name,
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "mAP50": f"{m['mAP50']:.4f}",
                "mAP50-95": f"{m['mAP50_95']:.4f}",
                "ms/frame": f"{m['inference_ms_per_image']:.3f}",
                "FPS": f"{m['fps']:.2f}",
            }
        )
    return rows


def _print_console_table(rows: List[Dict[str, str]]) -> None:
    headers = ["Model", "Precision", "Recall", "mAP50", "mAP50-95", "ms/frame", "FPS"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    def fmt(values: Dict[str, str]) -> str:
        return " | ".join(str(values[h]).ljust(widths[h]) for h in headers)

    line = "-+-".join("-" * widths[h] for h in headers)
    print(fmt({h: h for h in headers}))
    print(line)
    for row in rows:
        print(fmt(row))


def _save_csv(rows: List[Dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_graphs(rows: List[Dict[str, str]], output_dir: Path) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate graphs. Install with pip install matplotlib") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    models = [r["Model"] for r in rows]

    metric_keys = ["Precision", "Recall", "mAP50", "mAP50-95"]
    metric_values = {k: [float(r[k]) for r in rows] for k in metric_keys}
    speed_values = [float(r["ms/frame"]) for r in rows]
    fps_values = [float(r["FPS"]) for r in rows]

    perf_path = output_dir / "comparison_metrics.png"
    speed_path = output_dir / "comparison_speed.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for key, offset in zip(metric_keys, offsets):
        ax.bar([i + offset for i in x], metric_values[key], width=width, label=key)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv5 vs YOLO11 - Precision/Recall/mAP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(perf_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar([f"{m} (ms/frame)" for m in models], speed_values, label="ms/frame")
    ax2.set_ylabel("ms/frame")
    ax2.set_title("Inference Speed (Lower is better)")
    fig2.tight_layout()
    fig2.savefig(speed_path, dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(models, fps_values)
    ax3.set_ylabel("FPS")
    ax3.set_title("Inference Speed (Higher is better)")
    fig3.tight_layout()
    fps_path = output_dir / "comparison_fps.png"
    fig3.savefig(fps_path, dpi=150)
    plt.close(fig3)

    return {
        "metrics_chart": str(perf_path),
        "speed_chart_ms": str(speed_path),
        "speed_chart_fps": str(fps_path),
    }


def run_comparison(
    dataset_yaml: str,
    yolov5_weights: Optional[str] = None,
    yolo11_weights: Optional[str] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
    split: str = "test",
    save_csv: bool = False,
) -> Dict[str, object]:
    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    v5_weights = _resolve_yolov5_weights(yolov5_weights)
    v11_weights = resolve_model_path(DEFAULT_CONFIG, model_path=yolo11_weights)

    _delete_split_cache(dataset_path, split)
    yolo11_result = evaluate_detection_model(
            model_path=v11_weights,
            dataset_yaml=str(dataset_path),
            imgsz=imgsz,
            device=device,
            split=split,
        )
    _delete_split_cache(dataset_path, split)
    yolo5_result = evaluate_yolov5_native(
        dataset_yaml=str(dataset_path),
        model_path=v5_weights,
        imgsz=imgsz,
        device=device,
        batch_size=DEFAULT_CONFIG.batch,
    )
    results = {
        "YOLOv5": yolo5_result,
        "YOLO11": yolo11_result,
    }

    rows = _build_rows(results)
    _print_console_table(rows)

    output_dir = PROJECT_ROOT / "yolo11" / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "yolov5_vs_yolo11_comparison.json"
    charts = _save_graphs(rows, output_dir=output_dir)

    payload: Dict[str, object] = {
        "success": True,
        "dataset_yaml": str(dataset_path),
        "split": split,
        "imgsz": imgsz,
        "device": device or DEFAULT_CONFIG.device,
        "results": results,
        "comparison_table": rows,
        "charts": charts,
        "json_output_path": str(json_path),
    }

    if save_csv:
        csv_path = output_dir / "yolov5_vs_yolo11_comparison.csv"
        _save_csv(rows, csv_path)
        payload["csv_output_path"] = str(csv_path)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLOv5 and YOLO11 on precision, recall, mAP, and speed.")
    parser.add_argument("--data", default=str(DEFAULT_CONFIG.dataset_yaml), help="Path to dataset YAML.")
    parser.add_argument("--yolov5-weights", default=None, help="Path to YOLOv5 weights (best.pt).")
    parser.add_argument("--yolo11-weights", default=None, help="Path to YOLO11 weights (best.pt).")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--device", default=None, help="Device (cpu, cuda:0, etc.).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--save-csv", action="store_true", help="Also save comparison table as CSV.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_comparison(
        dataset_yaml=args.data,
        yolov5_weights=args.yolov5_weights,
        yolo11_weights=args.yolo11_weights,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        save_csv=args.save_csv,
    )
    print(json.dumps(result, indent=2))

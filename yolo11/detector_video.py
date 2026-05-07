import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

from yolo11.config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    ensure_dataset_class_order,
    expected_classes,
    resolve_model_path,
)
from yolo11.plate_ocr import read_plate_number

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "yolo11" / "outputs" / "output"
MIRROR_OUTPUT_DIR = PROJECT_ROOT / "yolo11" / "outputs" / "videos"
LEGACY_STATUS_TEXT = "Not"


def _status_to_string(value: Optional[bool]) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "None"


def _validate_model_classes(model: YOLO) -> None:
    names = model.names
    model_names = [str(names[idx]) for idx in sorted(names.keys())]
    expected = list(expected_classes(DEFAULT_CONFIG))
    if model_names != expected:
        raise ValueError(f"Model class names mismatch. model={model_names} expected={expected}")


def _prepare_legacy_output_dir(clear_existing: bool) -> None:
    LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not clear_existing:
        return
    for output_dir in (LEGACY_OUTPUT_DIR, MIRROR_OUTPUT_DIR):
        for old_file in output_dir.glob("Det_*.*"):
            if old_file.is_file():
                old_file.unlink()


def _write_legacy_txt(txt_path: Path, item: Dict[str, str]) -> None:
    txt_path.write_text(
        "\n".join(
            [
                item["image_path"],
                item["helmet_status"],
                item["lp_status"],
                item["plate_number"],
                item["num_passengers"],
                LEGACY_STATUS_TEXT,
            ]
        ),
        encoding="utf-8",
    )


def _write_dual_txt(filename: str, item: Dict[str, str]) -> None:
    _write_legacy_txt(LEGACY_OUTPUT_DIR / filename, item)
    _write_legacy_txt(MIRROR_OUTPUT_DIR / filename, item)


def _extract_best_lp_crop(prediction, frame):
    if prediction.boxes is None:
        return None

    best_conf = -1.0
    best_crop = None
    frame_h, frame_w = frame.shape[:2]

    for box in prediction.boxes:
        cls_id = int(box.cls.item())
        cls_name = prediction.names[cls_id]
        if cls_name != "LP":
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if conf > best_conf:
            best_conf = conf
            best_crop = crop

    return best_crop


def detect_video_yolov5_format(
    video_path: str,
    model_path: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Optional[str] = None,
    frame_stride: int = 20,
    clear_existing: bool = True,
) -> List[Dict[str, str]]:
    cfg = DEFAULT_CONFIG
    ensure_dataset_class_order(cfg)
    selected_model = resolve_model_path(cfg, model_path=model_path)
    selected_conf = conf if conf is not None else cfg.confidence
    selected_iou = iou if iou is not None else cfg.iou
    selected_device = device or cfg.device

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    model = YOLO(selected_model)
    _validate_model_classes(model)
    _prepare_legacy_output_dir(clear_existing=clear_existing)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")

    results: List[Dict[str, str]] = []
    frame_index = 0
    det_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            pred = model.predict(
                source=frame,
                conf=selected_conf,
                iou=selected_iou,
                device=selected_device,
                imgsz=cfg.imgsz,
                verbose=False,
                save=False,
            )[0]

            class_counts = {"Helmet": 0, "No Helmet": 0, "Rider": 0, "LP": 0}
            if pred.boxes is not None:
                for box in pred.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = pred.names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            useful = (
                int(class_counts.get("Rider", 0)) > 0
                or int(class_counts.get("No Helmet", 0)) > 0
                or int(class_counts.get("LP", 0)) > 0
            )
            if not useful:
                frame_index += 1
                continue

            no_helmet_count = int(class_counts.get("No Helmet", 0))
            helmet_count = int(class_counts.get("Helmet", 0))
            rider_count = int(class_counts.get("Rider", 0))
            lp_count = int(class_counts.get("LP", 0))

            if no_helmet_count > 0:
                helmet_status = False
            elif helmet_count > 0:
                helmet_status = True
            else:
                helmet_status = None

            plate_number = None
            if lp_count > 0:
                lp_crop = _extract_best_lp_crop(pred, frame)
                parsed_plate = read_plate_number(lp_crop, cfg) if lp_crop is not None else None
                lp_status = True if parsed_plate else False
                plate_number = parsed_plate
            else:
                lp_status = None

            image_name = f"Det_{det_index}.png"
            txt_name = f"Det_{det_index}.txt"
            image_out = LEGACY_OUTPUT_DIR / image_name
            mirror_image_out = MIRROR_OUTPUT_DIR / image_name
            plotted = pred.plot()
            if not cv2.imwrite(str(image_out), plotted):
                raise RuntimeError(f"Failed to save output image: {image_out}")
            if not cv2.imwrite(str(mirror_image_out), plotted):
                raise RuntimeError(f"Failed to save output image: {mirror_image_out}")

            item = {
                "image_path": str(image_out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "helmet_status": _status_to_string(helmet_status),
                "lp_status": _status_to_string(lp_status),
                "plate_number": str(plate_number or "None"),
                "num_passengers": str(rider_count),
            }
            _write_dual_txt(txt_name, item)
            results.append(item)

            det_index += 1
            frame_index += 1
    finally:
        cap.release()

    return results


def process_videos(
    input_path: str,
    model_path: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Optional[str] = None,
    frame_stride: int = 20,
    clear_existing: bool = True,
) -> Dict[str, object]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")

    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        files = [source]
    else:
        files = sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        if not files:
            raise FileNotFoundError(f"No videos found in folder: {source}")

    all_results: List[Dict[str, str]] = []
    for i, video_file in enumerate(files):
        all_results.extend(
            detect_video_yolov5_format(
                video_path=str(video_file),
                model_path=model_path,
                conf=conf,
                iou=iou,
                device=device,
                frame_stride=frame_stride,
                clear_existing=clear_existing and i == 0,
            )
        )

    return {"success": True, "results": all_results}


def detect_video(
    video_path: str,
    model_path: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Optional[str] = None,
    frame_stride: int = 20,
    clear_existing: bool = True,
) -> Dict[str, object]:
    return {
        "success": True,
        "results": detect_video_yolov5_format(
            video_path=video_path,
            model_path=model_path,
            conf=conf,
            iou=iou,
            device=device,
            frame_stride=frame_stride,
            clear_existing=clear_existing,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO11 video inference with YOLOv5-style outputs.")
    parser.add_argument("--video", default=None, help="Path to one video file (backward-compatible).")
    parser.add_argument("--input", default=None, help="Path to video file or video folder.")
    parser.add_argument("--model", default=None, help="Path to weights file. Defaults to trained weights.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold.")
    parser.add_argument("--device", default=None, help="Device (cpu, cuda:0, etc.).")
    parser.add_argument("--frame-stride", type=int, default=20, help="Process every Nth frame.")
    parser.add_argument("--keep-output", action="store_true", help="Do not clear existing Det_* files in output/.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    source = args.input or args.video
    if not source:
        raise ValueError("Provide --input <file_or_folder> or --video <file>.")

    response = process_videos(
        input_path=source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        frame_stride=args.frame_stride,
        clear_existing=not args.keep_output,
    )
    print(json.dumps(response, indent=2))

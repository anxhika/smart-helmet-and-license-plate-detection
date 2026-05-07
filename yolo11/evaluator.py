import time
import inspect
import types
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from ultralytics import YOLO

from yolo11.config import DEFAULT_CONFIG, resolve_model_path

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


def _patch_legacy_model_for_ultralytics(model: YOLO) -> None:
    raw_model = getattr(model, "model", None)
    if raw_model is None:
        return

    fuse_fn = getattr(raw_model, "fuse", None)
    if callable(fuse_fn):
        try:
            fuse_sig = inspect.signature(fuse_fn)
        except (TypeError, ValueError):
            fuse_sig = None
        if fuse_sig and "verbose" not in fuse_sig.parameters:
            original_fuse = fuse_fn

            def _fuse_compat(self, *args, **kwargs):
                kwargs.pop("verbose", None)
                return original_fuse(*args, **kwargs)

            raw_model.fuse = types.MethodType(_fuse_compat, raw_model)

    forward_fn = getattr(raw_model, "forward", None)
    if callable(forward_fn):
        try:
            forward_sig = inspect.signature(forward_fn)
        except (TypeError, ValueError):
            forward_sig = None
        if forward_sig and ("visualize" not in forward_sig.parameters or "embed" not in forward_sig.parameters):
            original_forward = forward_fn

            def _forward_compat(self, *args, **kwargs):
                kwargs.pop("visualize", None)
                kwargs.pop("embed", None)
                return original_forward(*args, **kwargs)

            raw_model.forward = types.MethodType(_forward_compat, raw_model)


def evaluate_detection_model(
    model_path: str,
    dataset_yaml: str,
    imgsz: int = 640,
    device: Optional[str] = None,
    split: str = "test",
) -> Dict[str, Any]:
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    selected_device = device or DEFAULT_CONFIG.device
    model = YOLO(model_path)
    _patch_legacy_model_for_ultralytics(model)
    val_results = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        device=selected_device,
        split=split,
        verbose=False,
    )
    box_metrics = val_results.box

    precision = float(getattr(box_metrics, "mp", 0.0))
    recall = float(getattr(box_metrics, "mr", 0.0))
    map50 = float(getattr(box_metrics, "map50", 0.0))
    map50_95 = float(getattr(box_metrics, "map", 0.0))

    speed_ms = dict(val_results.speed) if hasattr(val_results, "speed") else {}
    inference_ms = float(speed_ms.get("inference", 0.0))
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    return {
        "model_path": model_path,
        "dataset_yaml": str(Path(dataset_yaml)),
        "split": split,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map50_95,
            "inference_ms_per_image": inference_ms,
            "fps": fps,
        },
        "speed_ms": speed_ms,
    }


def evaluate_yolo11(
    dataset_yaml: str,
    model_path: Optional[str] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    selected_model = resolve_model_path(DEFAULT_CONFIG, model_path=model_path)
    return evaluate_detection_model(
        model_path=selected_model,
        dataset_yaml=dataset_yaml,
        imgsz=imgsz,
        device=device,
        split="test",
    )


def _process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    from utils.general import box_iou

    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.tensor(matches, device=iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def evaluate_yolov5_native(
    dataset_yaml: str,
    model_path: str,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    import yaml

    from models.experimental import attempt_load
    from utils.datasets import LoadImagesAndLabels
    from utils.general import box_iou, non_max_suppression, scale_coords, xywh2xyxy
    from utils.metrics import ap_per_class
    from utils.torch_utils import select_device

    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    test_path = data_cfg.get("test") or data_cfg.get("val")
    if not test_path:
        raise ValueError("dataset yaml must define 'test' or 'val' path")
    root_path = data_cfg.get("path", "")
    test_path = Path(test_path)
    if not test_path.is_absolute():
        if root_path:
            test_path = Path(root_path) / test_path
        else:
            test_path = Path(dataset_yaml).resolve().parent / test_path
    labels_guess = Path(str(test_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    cache_file = labels_guess.with_suffix(".cache")
    if cache_file.exists():
        cache_file.unlink()

    names_cfg = data_cfg.get("names", [])
    if isinstance(names_cfg, dict):
        names = [names_cfg[k] for k in sorted(names_cfg.keys(), key=lambda x: int(x))]
    else:
        names = list(names_cfg)
    nc = int(data_cfg.get("nc", len(names)))

    selected_device = select_device(device or DEFAULT_CONFIG.device)
    model = attempt_load(model_path, map_location=selected_device)
    stride = int(model.stride.max())
    imgsz = int(np.ceil(imgsz / stride) * stride)
    half = selected_device.type != "cpu"
    if half:
        model.half()
    model.eval()

    dataset = LoadImagesAndLabels(
        test_path,
        img_size=imgsz,
        batch_size=batch_size,
        augment=False,
        rect=True,
        single_cls=False,
        stride=stride,
        pad=0.5,
        prefix="val: ",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    iouv = torch.linspace(0.5, 0.95, 10, device=selected_device)
    niou = iouv.numel()
    stats = []
    seen = 0
    t_infer = 0.0
    t_nms = 0.0

    for img, targets, paths, shapes in dataloader:
        img = img.to(selected_device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(selected_device)

        t0 = time.perf_counter()
        out = model(img, augment=False)[0]
        t1 = time.perf_counter()
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
        t2 = time.perf_counter()
        t_infer += t1 - t0
        t_nms += t2 - t1

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:].clone()
            nl = labels.shape[0]
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if nl:
                labels[:, 1:5] *= torch.tensor(
                    [img[si].shape[2], img[si].shape[1], img[si].shape[2], img[si].shape[1]],
                    device=selected_device,
                )

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels[:, 0].cpu()))
                continue

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = _process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=selected_device)

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), torch.tensor(tcls)))

    if len(stats):
        stats_np = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats_np) and stats_np[0].any():
            p, r, ap, _, _ = ap_per_class(*stats_np, names=names)
            ap50 = ap[:, 0]
            ap5095 = ap.mean(1)
            mp = float(p.mean())
            mr = float(r.mean())
            map50 = float(ap50.mean())
            map5095 = float(ap5095.mean())
        else:
            mp = mr = map50 = map5095 = 0.0
    else:
        mp = mr = map50 = map5095 = 0.0

    infer_ms = (t_infer / max(seen, 1)) * 1000.0
    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0

    return {
        "model_path": str(Path(model_path)),
        "dataset_yaml": str(Path(dataset_yaml)),
        "split": "test",
        "metrics": {
            "precision": mp,
            "recall": mr,
            "mAP50": map50,
            "mAP50_95": map5095,
            "inference_ms_per_image": infer_ms,
            "fps": fps,
        },
        "speed_ms": {
            "inference": infer_ms,
            "nms": (t_nms / max(seen, 1)) * 1000.0,
        },
    }


def benchmark_inference_speed(
    image_paths: Iterable[str],
    model_path: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    selected_model = resolve_model_path(DEFAULT_CONFIG, model_path=model_path)
    selected_conf = conf if conf is not None else DEFAULT_CONFIG.confidence
    selected_iou = iou if iou is not None else DEFAULT_CONFIG.iou
    selected_device = device or DEFAULT_CONFIG.device

    model = YOLO(selected_model)
    timings_ms = []
    images_count = 0

    for image_path in image_paths:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        start = time.perf_counter()
        model.predict(
            source=str(image),
            conf=selected_conf,
            iou=selected_iou,
            device=selected_device,
            imgsz=DEFAULT_CONFIG.imgsz,
            verbose=False,
            save=False,
        )
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        images_count += 1

    if images_count == 0:
        raise ValueError("No images provided for speed benchmarking.")

    avg_ms = sum(timings_ms) / images_count
    return {
        "model_path": selected_model,
        "images_count": images_count,
        "avg_inference_ms": round(avg_ms, 3),
        "min_inference_ms": round(min(timings_ms), 3),
        "max_inference_ms": round(max(timings_ms), 3),
        "fps": round((1000.0 / avg_ms) if avg_ms > 0 else 0.0, 3),
    }

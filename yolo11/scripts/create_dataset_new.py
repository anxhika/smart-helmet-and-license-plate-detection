import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["Helmet", "No Helmet", "Rider", "LP"]


@dataclass(frozen=True)
class PairItem:
    source_split: str
    rel_no_ext: Path
    image_path: Path
    label_path: Path
    classes_present: Tuple[int, ...]


def read_classes(label_path: Path, nc: int) -> Tuple[int, ...]:
    classes: Set[int] = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        if 0 <= cls_id < nc:
            classes.add(cls_id)
    return tuple(sorted(classes))


def collect_pairs(source_root: Path, nc: int) -> List[PairItem]:
    pairs: List[PairItem] = []
    for split in SPLITS:
        img_dir = source_root / split / "images"
        lbl_dir = source_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for image_path in img_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTS:
                continue
            rel_no_ext = image_path.relative_to(img_dir).with_suffix("")
            label_path = lbl_dir / rel_no_ext.with_suffix(".txt")
            if not label_path.exists():
                continue
            classes_present = read_classes(label_path, nc=nc)
            if not classes_present:
                continue
            pairs.append(
                PairItem(
                    source_split=split,
                    rel_no_ext=rel_no_ext,
                    image_path=image_path,
                    label_path=label_path,
                    classes_present=classes_present,
                )
            )
    return pairs


def select_subset(
    pairs: List[PairItem],
    targets: Dict[str, int],
    nc: int,
    seed: int,
) -> Dict[str, List[PairItem]]:
    total_target = sum(targets.values())
    if len(pairs) < total_target:
        raise ValueError(f"Requested {total_target} samples but only {len(pairs)} labeled pairs are available.")

    all_class_presence = {c: 0 for c in range(nc)}
    for p in pairs:
        for c in p.classes_present:
            all_class_presence[c] += 1
    missing = [c for c, n in all_class_presence.items() if n == 0]
    if missing:
        raise ValueError(f"Missing classes in source dataset: {missing}")

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    total_target = sum(targets.values())
    class_target: Dict[str, Dict[int, int]] = {s: {} for s in SPLITS}
    for split in SPLITS:
        for cls_id in range(nc):
            proposed = round(all_class_presence[cls_id] * (targets[split] / total_target))
            class_target[split][cls_id] = max(1, proposed)

    selected: Dict[str, List[PairItem]] = {s: [] for s in SPLITS}
    class_counts: Dict[str, Counter] = {s: Counter() for s in SPLITS}
    used_ids: Set[Tuple[str, str]] = set()

    def can_add(split: str) -> bool:
        return len(selected[split]) < targets[split]

    def mark_used(split: str, item: PairItem) -> None:
        selected[split].append(item)
        for c in item.classes_present:
            class_counts[split][c] += 1
        used_ids.add((item.source_split, str(item.rel_no_ext)))

    # Seed each split with at least one sample for each class.
    for split in SPLITS:
        for cls_id in range(nc):
            if not can_add(split):
                break
            candidate = None
            for item in shuffled:
                item_id = (item.source_split, str(item.rel_no_ext))
                if item_id in used_ids or cls_id not in item.classes_present:
                    continue
                candidate = item
                break
            if candidate is not None:
                mark_used(split, candidate)

    # Fill remaining slots jointly so early splits do not consume all rare classes.
    while any(can_add(split) for split in SPLITS):
        split = max((s for s in SPLITS if can_add(s)), key=lambda s: targets[s] - len(selected[s]))
        best_item = None
        best_score = -1.0
        for item in shuffled:
            item_id = (item.source_split, str(item.rel_no_ext))
            if item_id in used_ids:
                continue
            score = 0.0
            for cls_id in item.classes_present:
                deficit = max(0, class_target[split][cls_id] - class_counts[split][cls_id])
                rarity_global = 1.0 / max(1, all_class_presence[cls_id])
                score += (3.0 * deficit) + (0.5 / (1 + class_counts[split][cls_id])) + (0.2 * rarity_global)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None:
            raise RuntimeError(f"Unable to complete split '{split}'.")
        mark_used(split, best_item)

    return selected


def copy_subset(selected: Dict[str, List[PairItem]], dest_root: Path, clean_dest: bool) -> None:
    if clean_dest:
        for split in SPLITS:
            for base in ("images", "labels"):
                split_dir = dest_root / base / split
                if split_dir.exists():
                    for p in split_dir.rglob("*"):
                        if p.is_file():
                            p.unlink()
                    for p in sorted(split_dir.rglob("*"), reverse=True):
                        if p.is_dir():
                            p.rmdir()

    for split in SPLITS:
        img_split_dir = dest_root / "images" / split
        lbl_split_dir = dest_root / "labels" / split
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        for item in selected[split]:
            # Flatten into split root to keep strict YOLO layout:
            # dataset_new/images/<split>/* and dataset_new/labels/<split>/*
            flat_stem = f"{item.source_split}__{'__'.join(item.rel_no_ext.parts)}"
            dest_img = img_split_dir / f"{flat_stem}{item.image_path.suffix}"
            dest_lbl = lbl_split_dir / f"{flat_stem}.txt"
            shutil.copy2(item.image_path, dest_img)
            shutil.copy2(item.label_path, dest_lbl)


def summarize(selected: Dict[str, List[PairItem]], nc: int) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        cls_counter = Counter()
        for item in selected[split]:
            for cls in item.classes_present:
                cls_counter[cls] += 1
        summary[split] = {"images": len(selected[split])}
        for cls_id in range(nc):
            summary[split][f"class_{cls_id}_{CLASS_NAMES[cls_id]}"] = cls_counter[cls_id]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create balanced dataset_new from existing YOLO dataset.")
    parser.add_argument("--source", default="dataset", help="Source dataset root (contains train/valid[/test]).")
    parser.add_argument("--dest", default="dataset_new", help="Destination dataset root.")
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--valid-count", type=int, default=70)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--clean-dest", action="store_true", help="Remove existing files under destination splits before copy.")
    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    targets = {"train": args.train_count, "valid": args.valid_count, "test": args.test_count}

    pairs = collect_pairs(source_root, nc=args.nc)
    selected = select_subset(pairs, targets=targets, nc=args.nc, seed=args.seed)
    copy_subset(selected, dest_root=dest_root, clean_dest=args.clean_dest)

    summary = summarize(selected, nc=args.nc)
    report_path = dest_root / "selection_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": summary, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

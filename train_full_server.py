from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve


CLASSES = [
    "Caption",
    "Footnote",
    "Form",
    "Key-Value Region",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
    "Document Index",
    "Formula",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PRETRAIN_URL = "https://zenodo.org/records/15881917/files/doclayout_yolo_indicdlp.pt?download=1"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = SCRIPT_DIR / "server_training_workdir"


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check)


def normalize_name(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").split()).lower()


CLASS_TO_ID = {normalize_name(name): index for index, name in enumerate(CLASSES)}


def configure_runtime(workdir: Path) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(workdir / "yolo_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")


def ensure_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    try:
        import doclayout_yolo  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "doclayout-yolo"])
    try:
        import yaml  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
    run([sys.executable, "-m", "pip", "uninstall", "-y", "wandb", "ray"], check=False)


def ensure_pretrain(pretrain: Path) -> Path:
    if pretrain.exists() and pretrain.stat().st_size > 1_000_000:
        print(f"Pretrained checkpoint exists: {pretrain}", flush=True)
        return pretrain

    pretrain.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained checkpoint to {pretrain}", flush=True)
    urlretrieve(PRETRAIN_URL, pretrain)
    return pretrain


def find_image(json_path: Path, image_name: str) -> Path | None:
    candidate = json_path.parent / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = json_path.parent / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    json_stem = json_path.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = json_path.parent / f"{json_stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def yolo_line(class_id: int, bbox: list[float], width: float, height: float) -> str | None:
    if len(bbox) != 4:
        return None
    x, y, box_width, box_height = bbox
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        return None

    x_center = (x + box_width / 2) / width
    y_center = (y + box_height / 2) / height
    norm_width = box_width / width
    norm_height = box_height / height

    values = [x_center, y_center, norm_width, norm_height]
    values = [min(1.0, max(0.0, value)) for value in values]
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def convert_one(json_path: Path) -> tuple[Path | None, list[str], Counter]:
    stats: Counter = Counter()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = data.get("images") or []
    if not images:
        stats["json_without_images"] += 1
        return None, [], stats

    image_info = images[0]
    image_name = image_info.get("file_name")
    width = float(image_info.get("width") or 0)
    height = float(image_info.get("height") or 0)
    if not image_name or width <= 0 or height <= 0:
        stats["bad_image_metadata"] += 1
        return None, [], stats

    image_path = find_image(json_path, image_name)
    if image_path is None:
        stats["missing_images"] += 1
        return None, [], stats

    categories = {
        category["id"]: category["name"]
        for category in data.get("categories", [])
        if "id" in category and "name" in category
    }

    lines: list[str] = []
    seen_lines: set[str] = set()
    for annotation in data.get("annotations", []):
        category_name = categories.get(annotation.get("category_id"))
        class_id = CLASS_TO_ID.get(normalize_name(category_name or ""))
        if class_id is None:
            stats[f"skipped_class:{category_name}"] += 1
            continue

        line = yolo_line(class_id, annotation.get("bbox", []), width, height)
        if line is None:
            stats["bad_boxes"] += 1
            continue
        if line in seen_lines:
            stats["duplicate_labels_removed"] += 1
            continue

        seen_lines.add(line)
        lines.append(line)
        stats[f"class:{CLASSES[class_id]}"] += 1

    return image_path, lines, stats


def write_data_yaml(output: Path) -> Path:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    text = f"""path: {output.resolve().as_posix()}
train: images/train
val: images/val

names:
{names}
"""
    data_yaml = output / "data.yaml"
    data_yaml.write_text(text, encoding="utf-8")
    return data_yaml


def prepare_existing_data_yaml(data_yaml_path: Path, workdir: Path) -> Path:
    """Use a dataset that has ALREADY been converted to YOLO format (e.g. one a
    teammate prepared and sent over). This skips JSON->YOLO conversion entirely
    and just makes sure the YAML's 'path' field points at the right folder on
    THIS machine (the YAML was likely generated on someone else's computer, so
    its original 'path' is almost certainly wrong here)."""
    import yaml

    data_yaml_path = data_yaml_path.resolve()
    if not data_yaml_path.exists():
        raise SystemExit(f"--data-yaml file not found: {data_yaml_path}")

    config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = data_yaml_path.parent

    # Force 'path' to the folder that actually contains this YAML on disk,
    # regardless of what was baked in when it was created elsewhere.
    config["path"] = dataset_root.resolve().as_posix()
    config.setdefault("train", "images/train")
    config.setdefault("val", "images/val")

    # Sanity-check that the referenced folders actually exist so we fail
    # loudly and clearly instead of letting the trainer error out later.
    for split_key in ("train", "val"):
        split_rel = config[split_key]
        split_path = dataset_root / split_rel
        if not split_path.exists():
            raise SystemExit(
                f"data.yaml says '{split_key}: {split_rel}' but "
                f"{split_path} does not exist. Open the dataset folder and "
                f"confirm the real subfolder names, then adjust the YAML."
            )

    workdir.mkdir(parents=True, exist_ok=True)
    fixed_path = workdir / "data_fixed.yaml"
    fixed_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Using existing dataset, corrected YAML written to: {fixed_path}", flush=True)
    print(f"  path -> {config['path']}", flush=True)
    print(f"  train -> {config['train']}", flush=True)
    print(f"  val   -> {config['val']}", flush=True)
    if "names" in config:
        print(f"  classes ({len(config['names'])}): {config['names']}", flush=True)
    return fixed_path


def convert_dataset(source: Path, output: Path, val_ratio: float, seed: int, overwrite: bool) -> Path:
    data_yaml = output / "data.yaml"
    if data_yaml.exists() and not overwrite:
        print(f"Using existing converted dataset: {data_yaml}", flush=True)
        return data_yaml

    if output.exists() and overwrite:
        shutil.rmtree(output)

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    json_files = sorted(source.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON annotations found under {source}")

    random.Random(seed).shuffle(json_files)
    val_count = round(len(json_files) * val_ratio)
    val_paths = set(json_files[:val_count])

    summary: Counter = Counter()
    written = 0
    for json_path in json_files:
        image_path, lines, stats = convert_one(json_path)
        summary.update(stats)
        if image_path is None:
            continue

        split = "val" if json_path in val_paths else "train"
        output_stem = f"{json_path.parent.name}_{image_path.stem}".replace(" ", "_")
        image_output = output / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
        label_output = output / "labels" / split / f"{output_stem}.txt"

        shutil.copy2(image_path, image_output)
        label_output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    data_yaml = write_data_yaml(output)
    summary["images_written"] = written
    summary["json_files_seen"] = len(json_files)

    summary_path = output / "conversion_summary.txt"
    summary_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in sorted(summary.items())) + "\n",
        encoding="utf-8",
    )

    print(f"Converted {written} images from {len(json_files)} JSON files.", flush=True)
    print(f"Dataset YAML: {data_yaml}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return data_yaml


def patch_process_runtime() -> None:
    import numpy as np
    import torch

    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid

    torch_load = torch.load

    def torch_load_compatible(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = torch_load_compatible

    try:
        import doclayout_yolo.utils.callbacks.wb as wb

        wb.callbacks.clear()
    except Exception:
        pass

    try:
        import doclayout_yolo.utils.callbacks.raytune as raytune

        raytune.callbacks.clear()
    except Exception:
        pass


def auto_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if count >= 2:
        return "0,1"
    if count == 1:
        return "0"
    return "cpu"


def train(args: argparse.Namespace, data_yaml: Path, pretrain: Path) -> None:
    patch_process_runtime()
    from doclayout_yolo import YOLOv10

    device = auto_device(args.device)
    print(f"Training device: {device}", flush=True)

    model = YOLOv10(str(pretrain))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        plots=args.plots,
        val=args.val,
        save_period=args.save_period,
        amp=args.amp,
        exist_ok=True,
    )


def verify_weights(project: Path, run_name: str) -> None:
    run_dir = project / run_name
    weights = sorted(run_dir.rglob("*.pt")) if run_dir.exists() else []
    if not weights:
        raise SystemExit(f"No weights found under {run_dir}")

    print("Saved weights:", flush=True)
    for weight in weights:
        print(f"  {weight} ({weight.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server training script for 14-class DocLayout-YOLO fine-tuning."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Folder containing raw image/COCO-json annotation pairs. Not needed if --data-yaml is used.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Path to an ALREADY-PREPARED YOLO-format dataset's data.yaml (e.g. one a teammate sent you, "
        "with images/ and labels/ folders already built). Skips JSON conversion entirely.",
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--pretrain", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, 0, 0,1, or cpu.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader worker processes. Default 0 because multiprocessing dataloaders have crashed "
        "on Windows before; raise to 2-4 only if training is stable and you want a speed boost.",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="doclayout_14class_full")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--val", action="store_true", help="Enable validation during training.")
    parser.add_argument("--plots", action="store_true", help="Enable plot generation during training.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP only if stable on the server.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.workdir = args.workdir.resolve()
    args.project = args.workdir / "runs"
    dataset_yolo = args.workdir / "dataset_yolo"
    pretrain = args.pretrain or (args.workdir / "doclayout_yolo_indicdlp.pt")

    if not args.data_yaml and not args.source:
        raise SystemExit(
            "Provide either --data-yaml (an already-prepared YOLO dataset) "
            "or --source (a folder of raw image/COCO-json pairs to convert)."
        )

    configure_runtime(args.workdir)
    ensure_dependencies(args.skip_install)
    pretrain = ensure_pretrain(pretrain)

    if args.data_yaml:
        data_yaml = prepare_existing_data_yaml(args.data_yaml, args.workdir)
    else:
        data_yaml = convert_dataset(
            source=args.source.resolve(),
            output=dataset_yolo,
            val_ratio=args.val_ratio,
            seed=args.seed,
            overwrite=args.overwrite_dataset,
        )
    train(args, data_yaml, pretrain)
    verify_weights(args.project, args.name)


if __name__ == "__main__":
    main()

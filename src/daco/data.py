"""File listing and filename parsing for the DCASE 2025 Task 2 dataset.

Canonical dev-set naming:
    section_00_<domain>_<split>_<condition>_<index>_<attributes...>.wav
e.g. section_00_source_train_normal_0001_car_A1_spd_28V.wav
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEV_MACHINES = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]

EVAL_MACHINES = [
    "AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
    "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar",
]


@dataclass
class Clip:
    path: Path
    machine: str
    section: str
    domain: str            # "source" | "target" | "unknown"
    split: str             # "train" | "test" | "supplemental" | "unknown"
    label: Optional[int]   # 0 = normal, 1 = anomaly, None = unlabeled


def parse_clip(path: Path, machine: str) -> Clip:
    tokens = path.stem.split("_")
    section = "unknown"
    domain = "unknown"
    split = "unknown"
    label: Optional[int] = None

    for i, tok in enumerate(tokens):
        if tok == "section" and i + 1 < len(tokens):
            section = tokens[i + 1]
        elif tok in ("source", "target"):
            domain = tok
        elif tok in ("train", "test", "supplemental"):
            split = tok
        elif tok == "normal":
            label = 0
        elif tok == "anomaly":
            label = 1

    # fall back to the parent directory name for the split (eval test files
    # carry no domain/condition tokens in their names)
    if split == "unknown" and path.parent.name in ("train", "test", "supplemental"):
        split = path.parent.name

    return Clip(path=path, machine=machine, section=section,
                domain=domain, split=split, label=label)


def list_clips(data_root: Path, machine: str,
               gt_root: Path | None = None) -> list[Clip]:
    """List clips for a machine; if gt_root points at the official
    dcase2025_task2_evaluator checkout, inject the post-challenge ground-truth
    anomaly labels and domains into the (otherwise unlabeled) eval test clips.
    """
    machine_dir = Path(data_root) / machine
    if not machine_dir.is_dir():
        raise FileNotFoundError(f"machine directory not found: {machine_dir}")
    wavs = sorted(machine_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav files under {machine_dir}")
    clips = [parse_clip(p, machine) for p in wavs]
    if gt_root is not None:
        gt = load_ground_truth(Path(gt_root), machine)
        n_hit = 0
        for c in clips:
            if c.split == "test" and c.path.name in gt:
                c.label, c.domain = gt[c.path.name]
                n_hit += 1
        n_test = sum(c.split == "test" for c in clips)
        if n_hit != n_test:
            raise RuntimeError(
                f"{machine}: ground truth covers {n_hit}/{n_test} test clips")
    return clips


def load_ground_truth(gt_root: Path, machine: str) -> dict[str, tuple[int, str]]:
    """filename -> (anomaly label, domain) from the official evaluator repo.

    CSV rows are `section_00_0000.wav,<v>`; in ground_truth_data v is
    0=normal/1=anomaly, in ground_truth_domain v is 0=source/1=target.
    """
    def read(sub: str) -> dict[str, int]:
        path = gt_root / sub / f"ground_truth_{machine}_section_00_test.csv"
        out = {}
        for line in path.read_text().strip().splitlines():
            name, v = line.rsplit(",", 1)
            out[name] = int(v)
        return out

    labels = read("ground_truth_data")
    domains = read("ground_truth_domain")
    if set(labels) != set(domains):
        raise RuntimeError(f"{machine}: label/domain ground truth file mismatch")
    return {n: (labels[n], "target" if domains[n] == 1 else "source")
            for n in labels}

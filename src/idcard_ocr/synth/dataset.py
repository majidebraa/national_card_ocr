"""Generate a labelled synthetic dataset (images + ``labels.jsonl``)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from idcard_ocr.layouts import LayoutRegistry
from idcard_ocr.synth.identity import random_identity
from idcard_ocr.synth.render import render_card
from idcard_ocr.synth.scene import DIFFICULTIES, compose_scene


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    file: str
    layout_id: str
    version: str
    side: str
    fields: dict[str, str]
    quad: list[list[float]]
    rotation: int
    difficulty: str


def generate_dataset(
    out_dir: str | Path,
    count: int,
    *,
    seed: int = 1234,
    difficulty: str = "medium",
    registry: LayoutRegistry | None = None,
    upside_down_fraction: float = 0.15,
) -> list[Sample]:
    """Write ``count`` scene photos round-robin over all layouts; deterministic for a seed."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    if count < 1:
        raise ValueError("count must be at least 1")
    reg = registry or LayoutRegistry.load()
    layouts = sorted(reg.all(), key=lambda lay: lay.id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    for i in range(count):
        rng = random.Random(f"{seed}:{i}")
        layout = layouts[i % len(layouts)]
        identity = random_identity(rng)
        flat = render_card(layout, identity, rng)
        scene = compose_scene(flat, rng, difficulty, rotate_180=rng.random() < upside_down_fraction)
        name = f"{i:05d}_{layout.id}.jpg"
        if not cv2.imwrite(str(out / name), scene.image):
            raise OSError(f"could not write {name}")
        truth = {k: v for k, v in identity.values().items() if k in layout.fields}
        samples.append(
            Sample(
                f"{i:05d}",
                name,
                layout.id,
                layout.version,
                layout.side,
                truth,
                scene.quad.round(1).tolist(),
                scene.rotation,
                difficulty,
            )
        )
    with (out / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return samples


def load_labels(directory: str | Path) -> list[Sample]:
    path = Path(directory) / "labels.jsonl"
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(Sample(**json.loads(line)))
    return samples

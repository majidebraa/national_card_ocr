#!/usr/bin/env python
"""Train the CRNN line recogniser (``idcard_ocr.ocr.crnn``) from labelled, rectified cards.

Needs PyTorch (``pip install torch onnx``); inference does not. Input is the card cache written by
``scripts/kaggle_eval.py --stage cache`` plus the label CSV. Stages, each resumable via ``--work``:

1. ``bootstrap`` - national-id only, nominal windows with jitter. Mis-registered cards teach noise
   here, so this model only has to be good enough for the next step.
2. ``align``     - label-verified pseudo-alignment: for every training card find the registration
   hypothesis under which the bootstrap model reads the *known* id correctly (nominal window first,
   then the grid). Only verified cards are used afterwards, and the dates inherit the same
   hypothesis, so their windows are aligned without any date model.
3. ``train``     - joint model for id + both dates on verified windows, warm-started from stage 1.
4. ``export``    - ONNX file for ``IDCARD_CRNN_MODEL`` (with a numerical parity check).

Example::

    python scripts/train_crnn.py bootstrap --cache data/cache --csv data/labels.csv --work work
    python scripts/train_crnn.py align     --cache data/cache --csv data/labels.csv --work work
    python scripts/train_crnn.py train     --cache data/cache --csv data/labels.csv --work work
    python scripts/train_crnn.py export    --work work --out crnn.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from idcard_ocr.layouts import CrnnSpec, parse_layout
from idcard_ocr.ocr.crnn import (
    BLANK,
    CHARSET,
    GRID,
    INPUT_H,
    INPUT_W,
    NOMINAL,
    centres,
    normalise,
    well_formed,
)

torch.set_num_threads(2)
LABEL_COLUMNS = {
    "national_id": "national_id",
    "birth_date": "date_of_birth",
    "expiry_date": "expiry_date",
}
Hyp = tuple[float, int, int]


class Net(nn.Module):
    """4 conv blocks -> vertical position kept in the channel axis -> 2x BiGRU -> CTC logits."""

    def __init__(self, ch: tuple[int, ...] = (32, 64, 96, 128), hidden: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = 2  # grayscale + a constant plane that tells the net which field it is reading
        for i, o in enumerate(ch):
            layers += [
                nn.Conv2d(c, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(True),
                nn.MaxPool2d(2 if i < 2 else (2, 1)),
            ]
            c = o
        self.cnn = nn.Sequential(*layers)
        self.mix = nn.Sequential(nn.Conv1d(c * 7, 256, 1), nn.ReLU(True), nn.Dropout(0.2))
        self.rnn = nn.GRU(
            256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(2 * hidden, len(CHARSET) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)
        b, c, h, w = f.shape
        return self.fc(self.rnn(self.mix(f.reshape(b, c * h, w)).transpose(1, 2))[0])


def to_tensor(windows: np.ndarray, planes: list[int]) -> torch.Tensor:
    x = torch.from_numpy(normalise(windows))[:, None]
    p = torch.tensor(planes, dtype=torch.float32).div(2.0).view(-1, 1, 1, 1).expand_as(x)
    return torch.cat([x, p], dim=1)


def aug_window(
    card: np.ndarray,
    centre: tuple[float, float],
    size: tuple[float, float],
    scale: float,
    rot: float,
    shear: float,
) -> np.ndarray:
    w, h = size[0] * scale, size[1] * scale
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    a = np.array([[c, -s + shear, 0], [s, c, 0], [0, 0, 1]], np.float32)
    t = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], np.float32)
    sc = np.array(
        [[INPUT_W / w, 0, INPUT_W / 2], [0, INPUT_H / h, INPUT_H / 2], [0, 0, 1]], np.float32
    )
    return cv2.warpAffine(
        card,
        (sc @ a @ t)[:2],
        (INPUT_W, INPUT_H),
        flags=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.4))
    out = (img.astype(np.float32) - 128) * rng.uniform(0.6, 1.3) + 128 + rng.uniform(-25, 25)
    if rng.random() < 0.5:
        out += rng.normal(0, rng.uniform(1, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def decode_batch(logits: torch.Tensor) -> tuple[list[str], list[float]]:
    p = F.softmax(logits, -1)
    top, best = p.max(-1)
    texts, confs = [], []
    for seq, tp in zip(best.tolist(), top.tolist(), strict=True):
        chars, probs, prev = [], [], -1
        for k, q in zip(seq, tp, strict=True):
            if k != prev and k != BLANK:
                chars.append(CHARSET[k])
                probs.append(q)
            prev = k
        texts.append("".join(chars))
        confs.append(float(np.mean(probs)) if probs else 0.0)
    return texts, confs


class Data:
    def __init__(self, args: argparse.Namespace) -> None:
        self.spec: CrnnSpec = parse_layout(json.loads(Path(args.layout).read_text())).crnn  # type: ignore[assignment]
        with Path(args.csv).open(encoding="utf-8-sig") as fh:
            self.rows = {int(r["id"]): r for r in csv.DictReader(fh)}
        cache = Path(args.cache)
        self.ids = [
            i for i in sorted(self.rows) if i % args.test_mod != 0 and (cache / f"{i}.png").exists()
        ]
        self.cards = {
            i: cv2.imread(str(cache / f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in self.ids
        }

    def label(self, i: int, field: str) -> str:
        return self.rows[i][LABEL_COLUMNS[field]].strip()


def run_epochs(
    net: nn.Module,
    data: Data,
    samples: list[tuple[int, str, Hyp]],
    epochs: int,
    lr: float,
    seed: int,
    per_epoch: int = 1376,
    negatives: float = 0.0,
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    steps = epochs * math.ceil(per_epoch / 32)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr * 1.5, total_steps=steps)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    spec, t0 = data.spec, time.time()
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(len(samples))[:per_epoch]
        total = 0.0
        for k in range(0, len(idx), 32):
            xs, planes, labels = [], [], []
            for j in idx[k : k + 32]:
                i, field, hyp = samples[j]
                cx, cy = centres(spec, hyp)[field]
                negative = field == "national_id" and rng.random() < negatives  # "no digits here"
                jy = rng.uniform(-10, 10) + (
                    rng.choice([-1, 1]) * rng.uniform(110, 135) if negative else 0
                )
                img = aug_window(
                    data.cards[i],
                    (cx + rng.uniform(-12, 12), cy + jy),
                    spec.window,
                    hyp[0] * rng.uniform(0.92, 1.1),
                    rng.uniform(-6, 6),
                    rng.uniform(-0.18, 0.18),
                )
                xs.append(photometric(img, rng))
                planes.append(spec.fields[field][2])
                labels.append("" if negative else data.label(i, field))
            logp = F.log_softmax(net(to_tensor(np.stack(xs), planes)), -1).transpose(0, 1)
            target = torch.tensor([CHARSET.index(c) for y in labels for c in y], dtype=torch.long)
            loss = ctc(
                logp,
                target,
                torch.full((len(labels),), logp.shape[0], dtype=torch.long),
                torch.tensor([len(y) for y in labels]),
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            sched.step()
            total += loss.item() * len(labels)
        if ep % 5 == 4 or ep == epochs - 1:
            print(
                f"epoch {ep + 1}/{epochs} loss {total / len(idx):.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )


def stage_bootstrap(args: argparse.Namespace) -> None:
    data = Data(args)
    samples = [(i, "national_id", NOMINAL) for i in data.ids]
    net = Net()
    run_epochs(net, data, samples, args.epochs, 2e-3, 0)
    torch.save(net.state_dict(), Path(args.work) / "boot.pt")


@torch.no_grad()
def stage_align(args: argparse.Namespace) -> None:
    data = Data(args)
    net = Net()
    net.load_state_dict(torch.load(Path(args.work) / "boot.pt"))
    net.eval()
    spec = data.spec
    verified: dict[int, Hyp] = {}

    def read(card: np.ndarray, hyps: list[Hyp]) -> list[tuple[str, float]]:
        wins = np.stack(
            [
                aug_window(card, centres(spec, h)["national_id"], spec.window, h[0], 0.0, 0.0)
                for h in hyps
            ]
        )
        texts, confs = decode_batch(net(to_tensor(wins, [0] * len(hyps))))
        return [
            (t, c if well_formed("national_id", t) else 0.0)
            for t, c in zip(texts, confs, strict=True)
        ]

    for n, i in enumerate(data.ids):
        want = data.label(i, "national_id")
        text, _ = read(data.cards[i], [NOMINAL])[0]
        hyp: Hyp | None = NOMINAL if text == want else None
        if hyp is None:
            best = max(
                zip(read(data.cards[i], list(GRID)), GRID, strict=True), key=lambda p: p[0][1]
            )
            hyp = best[1] if best[0][0] == want and best[0][1] > 0 else None
        if hyp is not None:
            verified[i] = hyp
        if n % 200 == 0:
            print(f"aligned {n}/{len(data.ids)}: {len(verified)} verified", flush=True)
    (Path(args.work) / "align.json").write_text(
        json.dumps({str(k): list(v) for k, v in verified.items()})
    )
    print(f"verified {len(verified)} of {len(data.ids)} training cards")


def stage_train(args: argparse.Namespace) -> None:
    data = Data(args)
    align = {
        int(k): (float(v[0]), int(v[1]), int(v[2]))
        for k, v in json.loads((Path(args.work) / "align.json").read_text()).items()
    }
    if args.no_align:  # ablation: every card at the nominal window, verified or not
        align = dict.fromkeys(data.ids, NOMINAL)
    samples = [(i, f, h) for i, h in align.items() for f in data.spec.fields]
    net = Net()
    net.load_state_dict(torch.load(Path(args.work) / "boot.pt"))
    # keep the id head and let the date classes (incl. "/") be learned from the verified windows
    run_epochs(net, data, samples, args.epochs, 1e-3, args.seed, negatives=0.05)
    torch.save(net.state_dict(), Path(args.work) / args.name)


def stage_export(args: argparse.Namespace) -> None:
    net = Net()
    net.load_state_dict(torch.load(Path(args.work) / args.name))
    net.eval()
    x = torch.randn(4, 2, INPUT_H, INPUT_W)
    torch.onnx.export(
        net,
        x,
        args.out,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(7, 2, INPUT_H, INPUT_W)
    with torch.no_grad():
        ref = net(probe).numpy()
    diff = float(np.abs(session.run(None, {"x": probe.numpy()})[0] - ref).max())
    print(
        f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB), max |diff| {diff:.2e}"
    )
    if diff > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["bootstrap", "align", "train", "export"])
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--csv", default="data/labels.csv")
    ap.add_argument("--layout", default="calibration/kaggle/new-front-kaggle.json")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="crnn.onnx")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1, help="seed of the joint stage")
    ap.add_argument("--name", default="joint.pt", help="checkpoint file written by `train`")
    ap.add_argument("--no-align", action="store_true", help="ablation: skip pseudo-alignment")
    ap.add_argument("--test-mod", type=int, default=5, help="ids divisible by this are held out")
    args = ap.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if args.epochs is None:
        args.epochs = {"bootstrap": 60, "train": 40}.get(args.stage, 0)
    {
        "bootstrap": stage_bootstrap,
        "align": stage_align,
        "train": stage_train,
        "export": stage_export,
    }[args.stage](args)


if __name__ == "__main__":
    main()

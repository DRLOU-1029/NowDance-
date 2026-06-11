from __future__ import annotations

import json
from pathlib import Path

from .schema import PoseSequence


def load_sequence(path: str | Path) -> PoseSequence:
    with Path(path).open("r", encoding="utf-8") as file:
        return PoseSequence.from_json(json.load(file))


def save_sequence(sequence: PoseSequence, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(sequence.to_json(), file, ensure_ascii=False, indent=2)


def save_json(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

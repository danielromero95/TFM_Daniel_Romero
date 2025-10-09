"""Minimal self-check for squat segmentation."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ml.segment import segment


def _mock_series(fps: int = 30, reps: int = 5, duration_s: float = 15.0) -> Dict[str, object]:
    frame_count = int(round(fps * duration_s))
    if frame_count <= 0:
        return {"fps": fps, "frames": []}

    t = np.arange(frame_count) / float(fps)
    frequency = reps / duration_s
    base = 1.0 + 0.3 * np.cos(2.0 * math.pi * frequency * t)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, size=frame_count)

    frames: List[Dict[str, object]] = []
    for value, jitter in zip(base, noise):
        left_y = float(value + jitter)
        right_y = float(value - jitter * 0.5)
        frames.append(
            {
                "landmarks": {
                    "LEFT_HIP": {"y": left_y},
                    "RIGHT_HIP": {"y": right_y},
                }
            }
        )

    return {"fps": fps, "frames": frames}


def run_selfcheck() -> None:
    cfg = {
        "segmentation": {
            "hip_min_drop_norm": 0.10,
            "peak_prominence": 0.05,
            "min_rep_duration_s": 1.0,
        }
    }
    series = _mock_series()
    windows = segment(series, cfg)
    assert len(windows) == 5, f"expected 5 reps, got {len(windows)}"


if __name__ == "__main__":
    run_selfcheck()
    print("OK")

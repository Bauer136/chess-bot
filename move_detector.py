"""Stage 3 v1: frame-difference temporal move detector.

State machine on the rectified board view from Stage 2:

    STABLE --(diff > disturbed_thresh)--> DISTURBED
    DISTURBED --(diff < stable_thresh for stable_frames consecutive frames)
        --> STABLE, emitting a MoveEvent(before, after).

Hysteresis between the two thresholds prevents flicker around the boundary.
This will misfire on shadows, knocked pieces, and clock hits — that's the
known limitation that motivates replacing it with C3D later.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class State(Enum):
    STABLE = "stable"
    DISTURBED = "disturbed"


@dataclass
class MoveEvent:
    index: int                # 1-based move counter within this run
    before: np.ndarray        # rectified BGR frame from before the disturbance
    after: np.ndarray         # rectified BGR frame after settling


class FrameDiffMoveDetector:
    def __init__(
        self,
        stable_thresh: float = 2.0,
        disturbed_thresh: float = 8.0,
        stable_frames: int = 15,
    ):
        if stable_thresh >= disturbed_thresh:
            raise ValueError("stable_thresh must be < disturbed_thresh (need hysteresis)")
        self.stable_thresh = stable_thresh
        self.disturbed_thresh = disturbed_thresh
        self.stable_frames = stable_frames
        self.reset()

    def reset(self) -> None:
        self.state = State.STABLE
        self.reference_gray: Optional[np.ndarray] = None
        self.reference_bgr: Optional[np.ndarray] = None
        self.before_frame: Optional[np.ndarray] = None
        self.stable_count = 0
        self.move_count = 0
        self.last_diff = 0.0

    def update(self, frame: np.ndarray) -> Optional[MoveEvent]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame seeds the reference.
        if self.reference_gray is None or self.reference_gray.shape != gray.shape:
            self.reference_gray = gray
            self.reference_bgr = frame.copy()
            return None

        diff = float(cv2.absdiff(gray, self.reference_gray).mean())
        self.last_diff = diff
        event: Optional[MoveEvent] = None

        if self.state is State.STABLE:
            if diff > self.disturbed_thresh:
                self.before_frame = self.reference_bgr
                self.state = State.DISTURBED
                self.stable_count = 0
        else:  # DISTURBED
            if diff < self.stable_thresh:
                self.stable_count += 1
                if self.stable_count >= self.stable_frames:
                    self.state = State.STABLE
                    self.move_count += 1
                    event = MoveEvent(self.move_count, self.before_frame, frame.copy())
                    self.reference_gray = gray
                    self.reference_bgr = frame.copy()
                    self.before_frame = None
            else:
                self.stable_count = 0

        return event

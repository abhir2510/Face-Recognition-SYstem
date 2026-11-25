"""Active liveness detection using MediaPipe facial landmarks (blink + head turn)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Sequence

# Standard 6-point MediaPipe FaceMesh indices for Eye Aspect Ratio (EAR).
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)

# Nose tip and outer cheek landmarks for horizontal head-turn (yaw) estimation.
NOSE_TIP_IDX = 1
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks: Sequence, indices: tuple[int, ...], width: int, height: int) -> float:
    """Compute Eye Aspect Ratio (EAR) in pixel space to stay aspect-ratio invariant."""
    pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
    vertical_1 = _dist(pts[1], pts[5])
    vertical_2 = _dist(pts[2], pts[4])
    horizontal = _dist(pts[0], pts[3])
    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(landmarks: Sequence, width: int, height: int) -> float:
    """Return mean EAR across both eyes."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, width, height)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, width, height)
    return 0.5 * (left + right)


def head_yaw_ratio(landmarks: Sequence) -> float:
    """Return normalized nose position between cheeks (~0.5 when facing forward)."""
    left_x = landmarks[LEFT_CHEEK_IDX].x
    right_x = landmarks[RIGHT_CHEEK_IDX].x
    nose_x = landmarks[NOSE_TIP_IDX].x
    lo, hi = min(left_x, right_x), max(left_x, right_x)
    span = max(hi - lo, 1e-6)
    return (nose_x - lo) / span


@dataclass
class LivenessResult:
    state: str  # "pending", "passed", "failed"
    prompt: str
    ear: float = 0.0
    yaw: float = 0.5


@dataclass
class LivenessChallenge:
    """Randomized active challenge-response liveness verifier."""

    steps: int = 2
    target_blinks: int = 2
    ear_threshold: float = 0.21
    blink_consec_frames: int = 2
    yaw_delta: float = 0.13
    center_tolerance: float = 0.08
    turn_consec_frames: int = 2
    timeout_seconds: float = 15.0

    _challenges: list[str] = field(default_factory=list, init=False)
    _step_idx: int = field(default=0, init=False)
    _state: str = field(default="pending", init=False)
    _started_at: float | None = field(default=None, init=False)
    _blinks: int = field(default=0, init=False)
    _closed_frames: int = field(default=0, init=False)
    _turn_frames: int = field(default=0, init=False)
    _turned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a fresh randomized challenge sequence."""
        pool = ["blink", "head_turn"]
        random.shuffle(pool)
        count = max(1, min(self.steps, len(pool)))
        self._challenges = pool[:count]
        self._step_idx = 0
        self._state = "pending"
        self._started_at = None
        self._reset_step_counters()

    def _reset_step_counters(self) -> None:
        self._blinks = 0
        self._closed_frames = 0
        self._turn_frames = 0
        self._turned = False

    def update(self, landmarks: Sequence, width: int, height: int, now: float | None = None) -> LivenessResult:
        """Advance the liveness state machine using the current face landmarks."""
        now = time.monotonic() if now is None else now
        ear = average_ear(landmarks, width, height)
        yaw = head_yaw_ratio(landmarks)

        if self._state == "passed":
            return LivenessResult(state="passed", prompt="Liveness verified", ear=ear, yaw=yaw)
        if self._state == "failed":
            return LivenessResult(state="failed", prompt="Liveness failed (timeout) - press r", ear=ear, yaw=yaw)

        if self._started_at is None:
            self._started_at = now
        elif now - self._started_at > self.timeout_seconds:
            self._state = "failed"
            return LivenessResult(state="failed", prompt="Liveness failed (timeout) - press r", ear=ear, yaw=yaw)

        challenge = self._challenges[self._step_idx]
        step_label = f"Liveness {self._step_idx + 1}/{len(self._challenges)}"

        if challenge == "blink":
            if ear < self.ear_threshold:
                self._closed_frames += 1
            else:
                if self._closed_frames >= self.blink_consec_frames:
                    self._blinks += 1
                self._closed_frames = 0

            if self._blinks >= self.target_blinks:
                self._advance_step()
            else:
                remaining = int(max(0, self.timeout_seconds - (now - self._started_at)))
                return LivenessResult(
                    state="pending",
                    prompt=f"{step_label}: Blink {self.target_blinks}x ({self._blinks}/{self.target_blinks}) [{remaining}s]",
                    ear=ear,
                    yaw=yaw,
                )

        elif challenge == "head_turn":
            if not self._turned:
                if abs(yaw - 0.5) >= self.yaw_delta:
                    self._turn_frames += 1
                    if self._turn_frames >= self.turn_consec_frames:
                        self._turned = True
                else:
                    self._turn_frames = 0
                if not self._turned:
                    remaining = int(max(0, self.timeout_seconds - (now - self._started_at)))
                    return LivenessResult(
                        state="pending",
                        prompt=f"{step_label}: Turn head slightly [{remaining}s]",
                        ear=ear,
                        yaw=yaw,
                    )

            # Once turned, wait for user to face forward again for clean recognition.
            if abs(yaw - 0.5) <= self.center_tolerance:
                self._advance_step()
            else:
                remaining = int(max(0, self.timeout_seconds - (now - self._started_at)))
                return LivenessResult(
                    state="pending",
                    prompt=f"{step_label}: Look back at camera [{remaining}s]",
                    ear=ear,
                    yaw=yaw,
                )

        if self._state == "passed":
            return LivenessResult(state="passed", prompt="Liveness verified", ear=ear, yaw=yaw)

        next_challenge = self._challenges[self._step_idx]
        next_hint = f"Blink {self.target_blinks}x" if next_challenge == "blink" else "Turn head slightly"
        return LivenessResult(
            state="pending",
            prompt=f"Liveness {self._step_idx + 1}/{len(self._challenges)}: {next_hint}",
            ear=ear,
            yaw=yaw,
        )

    def _advance_step(self) -> None:
        self._step_idx += 1
        self._reset_step_counters()
        if self._step_idx >= len(self._challenges):
            self._state = "passed"

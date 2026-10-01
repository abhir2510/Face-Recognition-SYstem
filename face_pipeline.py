"""Reusable detection, embedding, storage, and matching helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

EMBEDDINGS_FILE = Path("data/embeddings.npz")

class FacePipeline:
    """MTCNN detector plus pre-trained FaceNet embedding model."""

    def __init__(self) -> None:
        self.detector = MTCNN()
        self.embedder = FaceNet()

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return MTCNN detections after converting an OpenCV BGR frame to RGB."""
        return self.detector.detect_faces(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def crop_face(frame_bgr: np.ndarray, box: Iterable[int]) -> np.ndarray | None:
        x, y, width, height = (int(v) for v in box)
        x, y = max(0, x), max(0, y)
        right, bottom = min(frame_bgr.shape[1], x + width), min(frame_bgr.shape[0], y + height)
        if right <= x or bottom <= y:
            return None
        face = frame_bgr[y:bottom, x:right]
        return cv2.resize(face, (160, 160))

    def embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Make an L2-normalized 128-D FaceNet embedding from one cropped face."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        vector = self.embedder.embeddings(np.expand_dims(face_rgb, axis=0))[0]
        return vector / (np.linalg.norm(vector) + 1e-10)


def load_database(path: Path = EMBEDDINGS_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Load names and one averaged embedding per enrolled identity."""
    if not path.exists():
        # The column count is set by the installed FaceNet model on first enrollment.
        return np.array([], dtype=str), np.empty((0, 0), dtype=np.float32)
    stored = np.load(path, allow_pickle=False)
    return stored["names"], stored["embeddings"]


def save_identity(name: str, samples: list[np.ndarray], path: Path = EMBEDDINGS_FILE) -> None:
    """Replace or append a person's average embedding."""
    if not samples:
        raise ValueError("At least one embedding is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    names, embeddings = load_database(path)
    average = np.mean(np.stack(samples), axis=0)
    average /= np.linalg.norm(average) + 1e-10
    keep = names != name
    updated_embeddings = np.vstack((embeddings[keep], average)) if len(embeddings) else average[None, :]
    np.savez_compressed(path, names=np.append(names[keep], name), embeddings=updated_embeddings)


def best_match(query: np.ndarray, names: np.ndarray, embeddings: np.ndarray, threshold: float) -> tuple[str, float]:
    """Return the closest name by cosine similarity, or Unknown below threshold."""
    if len(names) == 0:
        return "Unknown", 0.0
    scores = embeddings @ query  # both vectors are L2-normalized
    index = int(np.argmax(scores))
    score = float(scores[index])
    return (str(names[index]) if score >= threshold else "Unknown"), score

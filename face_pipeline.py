"""Reusable detection, embedding, storage, and matching helpers."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet

EMBEDDINGS_FILE = Path("data/embeddings.npz")
DEFAULT_LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_landmarker_model(path: Path = DEFAULT_LANDMARKER_MODEL) -> Path:
    """Download the MediaPipe FaceLandmarker model if it is not present locally."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face landmarker model to {path}...")
    urllib.request.urlretrieve(LANDMARKER_MODEL_URL, path)
    return path


def _landmarks_to_box(landmarks: list, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Compute a tight face bounding box from normalized MediaPipe landmarks."""
    xs = [lm.x * frame_width for lm in landmarks]
    ys = [lm.y * frame_height for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y
    pad_x = 0.05 * box_w
    pad_top = 0.08 * box_h
    pad_bottom = 0.03 * box_h

    x1 = max(0, int(round(min_x - pad_x)))
    y1 = max(0, int(round(min_y - pad_top)))
    x2 = min(frame_width, int(round(max_x + pad_x)))
    y2 = min(frame_height, int(round(max_y + pad_bottom)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


class FacePipeline:
    """Face detector (MediaPipe FaceLandmarker or MTCNN) plus FaceNet embedder."""

    def __init__(
        self,
        detector: str = "mediapipe",
        video_mode: bool = False,
        max_faces: int = 3,
        min_detection_confidence: float = 0.6,
        landmarker_model_path: Path = DEFAULT_LANDMARKER_MODEL,
    ) -> None:
        if detector not in {"mediapipe", "mtcnn"}:
            raise ValueError(f"Unsupported detector: {detector!r}. Use 'mediapipe' or 'mtcnn'.")
        self.detector_name = detector
        self.video_mode = video_mode
        self._last_timestamp_ms = -1
        self._landmarker = None
        self._mtcnn = None

        if detector == "mediapipe":
            model_path = ensure_landmarker_model(landmarker_model_path)
            running_mode = (
                mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=running_mode,
                num_faces=max_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        else:
            from mtcnn import MTCNN

            self._mtcnn = MTCNN()

        self.embedder = FaceNet()
        image_size = self.embedder.metadata["image_size"]

        @tf.function(reduce_retracing=True)
        def _infer(batch: tf.Tensor) -> tf.Tensor:
            return self.embedder.model(batch, training=False)

        self._infer_embedding = _infer
        # Warm up TensorFlow graph tracing once at startup so the first live frame does not stutter.
        self._infer_embedding(tf.zeros((1, image_size, image_size, 3), dtype=tf.float32))

    def close(self) -> None:
        """Release detector resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> FacePipeline:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[dict]:
        """Detect faces and return boxes (plus MediaPipe landmarks when available)."""
        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.detector_name == "mtcnn":
            assert self._mtcnn is not None
            return self._mtcnn.detect_faces(frame_rgb)

        assert self._landmarker is not None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.video_mode:
            now_ms = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
            ts = max(self._last_timestamp_ms + 1, now_ms)
            self._last_timestamp_ms = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)

        height, width = frame_bgr.shape[:2]
        detections = []
        for face_landmarks in result.face_landmarks:
            box = _landmarks_to_box(face_landmarks, width, height)
            if box[2] > 0 and box[3] > 0:
                detections.append(
                    {
                        "box": box,
                        "confidence": 1.0,
                        "landmarks": face_landmarks,
                    }
                )
        return detections

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
        """Make an L2-normalized 512-D FaceNet embedding from one cropped face."""
        target_size = self.embedder.metadata["image_size"]
        if face_bgr.shape[0] != target_size or face_bgr.shape[1] != target_size:
            face_bgr = cv2.resize(face_bgr, (target_size, target_size))
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        normalized = np.expand_dims(self.embedder._normalize(face_rgb), axis=0).astype(np.float32)
        vector = self._infer_embedding(tf.constant(normalized))[0].numpy()
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

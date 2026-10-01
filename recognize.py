"""Recognize enrolled, consented faces from a webcam stream."""
from __future__ import annotations

import argparse
import time

import cv2

from face_pipeline import FacePipeline, best_match, load_database

def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize faces from the webcam.")
    parser.add_argument("--threshold", type=float, default=0.72, help="Cosine similarity threshold (default: 0.72)")
    args = parser.parse_args()
    names, database = load_database()
    if len(names) == 0:
        raise RuntimeError("No enrolled identities. Run enroll.py first.")
    pipeline = FacePipeline()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0.")

    previous_time = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            for detection in pipeline.detect(frame):
                if detection["confidence"] < 0.95:
                    continue
                face = pipeline.crop_face(frame, detection["box"])
                if face is None:
                    continue
                name, score = best_match(pipeline.embedding(face), names, database, args.threshold)
                x, y, width, height = map(int, detection["box"])
                colour = (0, 210, 0) if name != "Unknown" else (0, 100, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
            now = time.time()
            fps = 1 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(frame, f"FPS: {fps:.1f} | q: quit", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Facial Recognition - Consent-based Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 

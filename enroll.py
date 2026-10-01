"""Capture consented webcam samples and save a FaceNet identity embedding."""
from __future__ import annotations

import argparse
import time

import cv2

from face_pipeline import FacePipeline, save_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a consenting person from a webcam.")
    parser.add_argument("--name", required=True, help="Label to save, for example Abhishek")
    parser.add_argument("--samples", type=int, default=30, help="Number of good face samples to collect")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    pipeline = FacePipeline()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0. Check the camera permission or use another index.")

    samples, last_capture = [], 0.0
    print("Press c to capture a detected face. Press q to cancel.")
    try:
        while len(samples) < args.samples:
            ok, frame = camera.read()
            if not ok:
                break
            detections = pipeline.detect(frame)
            usable = max(detections, key=lambda item: item["confidence"], default=None)
            if usable and usable["confidence"] >= 0.95:
                x, y, width, height = map(int, usable["box"])
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
                cv2.putText(frame, "Face ready - press c", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.putText(frame, f"{args.name}: {len(samples)}/{args.samples}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Enrollment (c = capture, q = quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord("c") and usable and time.time() - last_capture > 0.4:
                face = pipeline.crop_face(frame, usable["box"])
                if face is not None:
                    samples.append(pipeline.embedding(face))
                    last_capture = time.time()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if samples:
        save_identity(args.name, samples)
        print(f"Saved {len(samples)} samples for {args.name}.")


if __name__ == "__main__":
    main()

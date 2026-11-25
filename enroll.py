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
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--detector",
        choices=("mediapipe", "mtcnn"),
        default="mediapipe",
        help="Face detector backend (default: mediapipe)",
    )
    parser.add_argument("--auto", action="store_true", help="Automatically capture samples every 0.35s")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}. Check the camera permission or use another index.")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    min_conf = 0.5 if args.detector == "mediapipe" else 0.95
    samples, last_capture = [], 0.0
    print("Press c to capture a detected face (or run with --auto). Press q to cancel.")

    with FacePipeline(detector=args.detector, video_mode=True, max_faces=1) as pipeline:
        try:
            while len(samples) < args.samples:
                ok, frame = camera.read()
                if not ok:
                    break
                detections = pipeline.detect(frame)
                usable = max(detections, key=lambda item: item["confidence"], default=None)
                ready = usable is not None and usable["confidence"] >= min_conf
                if ready:
                    x, y, width, height = map(int, usable["box"])
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
                    hint = "Auto-capturing..." if args.auto else "Face ready - press c"
                    cv2.putText(frame, hint, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    f"{args.name}: {len(samples)}/{args.samples}",
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Enrollment (c = capture, q = quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return
                now = time.monotonic()
                should_capture = (args.auto and now - last_capture > 0.35) or (
                    key == ord("c") and now - last_capture > 0.35
                )
                if ready and should_capture:
                    assert usable is not None
                    face = pipeline.crop_face(frame, usable["box"])
                    if face is not None:
                        samples.append(pipeline.embedding(face))
                        last_capture = now
        finally:
            camera.release()
            cv2.destroyAllWindows()

    if samples:
        save_identity(args.name, samples)
        print(f"Saved {len(samples)} samples for {args.name}.")


if __name__ == "__main__":
    main()

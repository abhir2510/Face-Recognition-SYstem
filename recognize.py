"""Recognize enrolled, consented faces from a webcam stream with active liveness."""
from __future__ import annotations

import argparse
from collections import deque
import time

import cv2

from face_pipeline import FacePipeline, best_match, load_database
from liveness import LivenessChallenge


def _draw_label(
    frame: cv2.Mat,
    box: tuple[int, int, int, int],
    text: str,
    colour: tuple[int, int, int],
) -> None:
    x, y, width, height = box
    cv2.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
    cv2.putText(frame, text, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize faces from the webcam with liveness check.")
    parser.add_argument("--threshold", type=float, default=0.72, help="Cosine similarity threshold (default: 0.72)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--detector",
        choices=("mediapipe", "mtcnn"),
        default="mediapipe",
        help="Face detector backend (default: mediapipe)",
    )
    parser.add_argument(
        "--recognition-interval",
        type=int,
        default=5,
        help="Run FaceNet every N frames and cache result between frames (default: 5)",
    )
    parser.add_argument("--no-liveness", action="store_true", help="Disable active blink/head-turn liveness check")
    parser.add_argument(
        "--liveness-steps",
        type=int,
        default=2,
        choices=(1, 2),
        help="Number of randomized liveness challenges (default: 2)",
    )
    parser.add_argument(
        "--liveness-timeout",
        type=float,
        default=15.0,
        help="Seconds allowed to complete liveness challenge (default: 15.0)",
    )
    args = parser.parse_args()
    if args.recognition_interval < 1:
        parser.error("--recognition-interval must be at least 1")

    names, database = load_database()
    if len(names) == 0:
        raise RuntimeError("No enrolled identities. Run enroll.py first.")

    use_liveness = not args.no_liveness
    if use_liveness and args.detector != "mediapipe":
        print("Warning: EAR/head-turn liveness requires MediaPipe landmarks; disabling liveness for MTCNN.")
        use_liveness = False

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}.")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    liveness = LivenessChallenge(steps=args.liveness_steps, timeout_seconds=args.liveness_timeout)
    frame_times: deque[float] = deque(maxlen=15)
    last_face_seen = time.monotonic()
    frame_index = 0
    cached_match: tuple[str, float] | None = None
    cached_multi: list[tuple[tuple[int, int, int, int], str, float]] = []

    with FacePipeline(detector=args.detector, video_mode=True, max_faces=3) as pipeline:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break

                now = time.monotonic()
                frame_times.append(now)
                height, width = frame.shape[:2]
                detections = pipeline.detect(frame)

                if use_liveness:
                    usable = [d for d in detections if d.get("landmarks") is not None]
                    primary = max(usable, key=lambda d: d["box"][2] * d["box"][3], default=None)

                    if primary is None:
                        if now - last_face_seen > 1.5:
                            liveness.reset()
                            cached_match = None
                            frame_index = 0
                    else:
                        last_face_seen = now
                        box = tuple(map(int, primary["box"]))
                        live = liveness.update(primary["landmarks"], width, height, now=now)

                        if live.state == "pending":
                            cached_match = None
                            _draw_label(frame, box, live.prompt, (0, 215, 255))
                        elif live.state == "failed":
                            cached_match = None
                            _draw_label(frame, box, live.prompt, (0, 0, 255))
                        else:
                            if cached_match is None or frame_index % args.recognition_interval == 0:
                                face = pipeline.crop_face(frame, box)
                                if face is not None:
                                    cached_match = best_match(pipeline.embedding(face), names, database, args.threshold)
                            frame_index += 1
                            if cached_match is not None:
                                name, score = cached_match
                                colour = (0, 210, 0) if name != "Unknown" else (0, 120, 255)
                                _draw_label(frame, box, f"{name} ({score:.2f}) [LIVE]", colour)

                        cv2.putText(
                            frame,
                            f"EAR: {live.ear:.2f} | Yaw: {live.yaw:.2f}",
                            (15, height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (220, 220, 220),
                            1,
                        )
                else:
                    min_conf = 0.5 if args.detector == "mediapipe" else 0.95
                    valid = [d for d in detections if d["confidence"] >= min_conf]
                    if frame_index % args.recognition_interval == 0 or not cached_multi:
                        updated: list[tuple[tuple[int, int, int, int], str, float]] = []
                        for det in valid:
                            box = tuple(map(int, det["box"]))
                            face = pipeline.crop_face(frame, box)
                            if face is None:
                                continue
                            name, score = best_match(pipeline.embedding(face), names, database, args.threshold)
                            updated.append((box, name, score))
                        cached_multi = updated
                    elif valid and len(valid) == len(cached_multi):
                        # Update boxes smoothly between FaceNet runs.
                        cached_multi = [
                            (tuple(map(int, det["box"])), prev[1], prev[2])
                            for det, prev in zip(valid, cached_multi)
                        ]
                    elif not valid:
                        cached_multi = []
                    frame_index += 1

                    for box, name, score in cached_multi:
                        colour = (0, 210, 0) if name != "Unknown" else (0, 100, 255)
                        _draw_label(frame, box, f"{name} ({score:.2f})", colour)

                fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6) if len(frame_times) >= 2 else 0.0
                controls = "q: quit | r: reset liveness" if use_liveness else "q: quit"
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | {controls}",
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Facial Recognition - Consent-based Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and use_liveness:
                    liveness.reset()
                    cached_match = None
                    frame_index = 0
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

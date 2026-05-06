"""Chess-bot pipeline entry point.

Stages 1-3 (v1): capture frames, apply the saved perspective warp + Gaussian
blur, push rectified frames onto a queue. A consumer thread runs the
frame-difference move detector; each detected move emits before/after JPEGs
and a console line. C3D will replace the detector later as a drop-in.

Run calibrate.py first to produce calibration.json. Without it, frames pass
through unrectified.

Hotkeys (display window focused):
    q = quit
    c = recalibrate corners
    r = reset to raw camera view (drop the in-memory warp matrix)
"""

import argparse
import json
import queue
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibrate import click_corners, save_calibration
from move_detector import FrameDiffMoveDetector, MoveEvent, State

SENTINEL = None
CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess-bot capture loop")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--queue-size",
        type=int,
        default=4,
        help="Frames buffered between capture and consumer. Small = drop stale frames.",
    )
    parser.add_argument(
        "--blur-ksize",
        type=int,
        default=5,
        help="Gaussian blur kernel size (must be odd). 0 disables blur.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the cv2.imshow debug window (e.g. headless runs).",
    )
    parser.add_argument(
        "--stable-thresh",
        type=float,
        default=2.0,
        help="Mean abs pixel diff below this counts as 'stable'.",
    )
    parser.add_argument(
        "--disturbed-thresh",
        type=float,
        default=8.0,
        help="Mean abs pixel diff above this triggers DISTURBED.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=15,
        help="Consecutive stable frames required to settle and fire a move event.",
    )
    parser.add_argument(
        "--moves-dir",
        default="moves",
        help="Directory to save before/after JPEGs of detected moves.",
    )
    parser.add_argument(
        "--no-save-moves",
        action="store_true",
        help="Skip writing move JPEGs to disk.",
    )
    return parser.parse_args()


def build_transform(corners, board_size: int):
    src = np.array(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def load_calibration():
    """Return (transform_matrix, board_size) or (None, None) if uncalibrated."""
    if not CALIBRATION_PATH.exists():
        print(f"[warn] {CALIBRATION_PATH.name} not found — run calibrate.py first.")
        return None, None
    payload = json.loads(CALIBRATION_PATH.read_text())
    size = int(payload["board_size"])
    return build_transform(payload["corners"], size), size


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def preprocess(frame, matrix, board_size: int, blur_ksize: int):
    """Warp the frame to a top-down square and apply Gaussian blur.

    Returns uint8 BGR. RGB conversion + float normalization are deferred to
    the model-input step so segmentation (Stage 4) and the imshow debug view
    keep working on cheap uint8 buffers.
    """
    if matrix is not None:
        out = cv2.warpPerspective(frame, matrix, (board_size, board_size))
    else:
        out = frame
    if blur_ksize and blur_ksize >= 3:
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)
    return out


def handle_move(event: MoveEvent, moves_dir: Optional[Path]) -> None:
    print(f"[move {event.index:03d}] detected")
    if moves_dir is None:
        return
    moves_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(moves_dir / f"move_{event.index:03d}_before.jpg"), event.before)
    cv2.imwrite(str(moves_dir / f"move_{event.index:03d}_after.jpg"), event.after)


def consumer(
    frame_q: "queue.Queue",
    stop: threading.Event,
    detector: FrameDiffMoveDetector,
    moves_dir: Optional[Path],
) -> None:
    while not stop.is_set():
        item = frame_q.get()
        if item is SENTINEL:
            break
        event = detector.update(item)
        if event is not None:
            handle_move(event, moves_dir)


def draw_overlay(view, detector: FrameDiffMoveDetector) -> None:
    color = (0, 200, 0) if detector.state is State.STABLE else (0, 0, 255)
    cv2.circle(view, (20, 22), 8, color, -1)
    text = f"diff={detector.last_diff:5.2f}  state={detector.state.value}  moves={detector.move_count}"
    cv2.putText(view, text, (38, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> None:
    args = parse_args()
    matrix, board_size = load_calibration()
    if board_size is None:
        board_size = 512  # fallback when uncalibrated

    cap = open_capture(args.source, args.width, args.height, args.fps)

    detector = FrameDiffMoveDetector(
        stable_thresh=args.stable_thresh,
        disturbed_thresh=args.disturbed_thresh,
        stable_frames=args.stable_frames,
    )
    moves_dir: Optional[Path] = None if args.no_save_moves else Path(args.moves_dir)

    frame_q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop = threading.Event()
    worker = threading.Thread(
        target=consumer, args=(frame_q, stop, detector, moves_dir), daemon=True
    )
    worker.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rectified = preprocess(frame, matrix, board_size, args.blur_ksize)

            try:
                frame_q.put_nowait(rectified)
            except queue.Full:
                # Drop the oldest frame so the consumer always sees fresh data.
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
                frame_q.put_nowait(rectified)

            if not args.no_display:
                view = rectified.copy()  # don't mutate the frame the consumer is reading
                draw_overlay(view, detector)
                cv2.imshow("chess-bot board", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    cv2.destroyWindow("chess-bot board")
                    pts = click_corners(frame)
                    if pts is None:
                        print("[calibrate] aborted, keeping previous calibration")
                    else:
                        save_calibration(pts, board_size, (frame.shape[1], frame.shape[0]))
                        matrix = build_transform(pts, board_size)
                        detector.reset()
                        print("[calibrate] new calibration saved, detector reset")
                if key == ord("r"):
                    matrix = None
                    cv2.destroyWindow("chess-bot board")
                    detector.reset()
                    print("[calibrate] reset to raw camera view, detector reset")
    finally:
        stop.set()
        frame_q.put(SENTINEL)
        worker.join(timeout=2.0)
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

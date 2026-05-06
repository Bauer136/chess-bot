"""One-time perspective calibration.

Grabs a single frame from the configured source, lets you click the four
board corners in TL -> TR -> BR -> BL order, and writes the result to
calibration.json next to this script. main.py loads that file at startup.

Keys: r = reset clicks, Enter/Space = confirm, q = abort.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
OUT_PATH = Path(__file__).parent / "calibration.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="0", help="Camera index or video path.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--board-size",
        type=int,
        default=512,
        help="Side length (px) of the rectified square board image.",
    )
    return p.parse_args()


def grab_frame(source: str, width: int, height: int):
    src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Discard a few frames so auto-exposure settles.
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from source.")
    return frame


def click_corners(frame):
    points: list[tuple[int, int]] = []
    window = "calibrate"

    def render():
        view = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(view, p, 8, (0, 255, 0), 2)
            cv2.putText(
                view, CORNER_LABELS[i], (p[0] + 12, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        if len(points) < 4:
            prompt = f"click {CORNER_LABELS[len(points)]}"
        else:
            prompt = "Enter to confirm, r to reset, q to abort"
        cv2.putText(
            view, prompt, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.imshow(window, view)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            render()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    render()
    try:
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"):
                return None
            if k == ord("r"):
                points.clear()
                render()
            if len(points) == 4 and k in (13, 32):  # Enter or Space
                return points
    finally:
        cv2.destroyWindow(window)


def save_calibration(corners, board_size: int, frame_size) -> Path:
    payload = {
        "corners": [[int(x), int(y)] for x, y in corners],
        "board_size": int(board_size),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    return OUT_PATH


def main() -> None:
    args = parse_args()
    frame = grab_frame(args.source, args.width, args.height)
    pts = click_corners(frame)
    if pts is None:
        print("calibration aborted")
        sys.exit(1)
    out = save_calibration(pts, args.board_size, (frame.shape[1], frame.shape[0]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

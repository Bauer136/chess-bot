"""Perspective calibration helpers, invoked from main.py via the 'c' hotkey.

main.py owns the camera and the live capture loop. When the user presses
'c' on the debug window, main.py hands the current frame to `click_corners`
here, then writes the result with `save_calibration`. This module is
library-only — running `python calibrate.py` directly is no longer the
calibration path.

UI keys inside the calibration window:
    left-click  pick the next corner (TL -> TR -> BR -> BL)
    r           reset clicks
    Enter/Space confirm
    q           abort
"""

import json
from pathlib import Path

import cv2

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
OUT_PATH = Path(__file__).parent / "calibration.json"


def click_corners(frame):
    """Open an interactive window over `frame` and return the 4 clicked corners.

    Returns the corners in TL -> TR -> BR -> BL order, or None if the user
    aborted with 'q'.
    """
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

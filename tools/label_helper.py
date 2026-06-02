"""Keyboard-driven crop labeller for the CNN training set.

Run with a directory of unlabeled square crops (e.g. the output of
`python main.py --save-squares` under `moves/move_NNN_squares/`); the
tool shows each crop upscaled and waits for a single keystroke that
moves the file into the matching folder under `data/`.

    python tools/label_helper.py moves/move_001_squares/

Keymap (one keystroke per crop):
    space or .     empty
    p n b r q k    black pieces
    P N B R Q K    white pieces (hold shift)
    s              skip (leave the file in place, advance)
    u              undo (move the last labelled file back, rewind)
    Esc            quit

Filenames in `data/<class>/` get prefixed with the source directory's
name so crops from different positions don't collide on `a8.jpg`.
"""

import argparse
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DISPLAY_SCALE = 8

# Folder name lookup: the empty-square label is `.`, but `.` is not a
# friendly directory name, so it lives under `empty/`.
CLASSES = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def folder_for(label: str) -> str:
    return "empty" if label == "." else label


# Single-keystroke -> label. Capital letters arrive as their own ASCII
# codes when the user holds shift, so we list both cases explicitly.
KEYMAP = {
    ord(" "): ".",
    ord("."): ".",
    ord("p"): "p", ord("n"): "n", ord("b"): "b",
    ord("r"): "r", ord("q"): "q", ord("k"): "k",
    ord("P"): "P", ord("N"): "N", ord("B"): "B",
    ord("R"): "R", ord("Q"): "Q", ord("K"): "K",
}
SKIP_KEYS = {ord("s")}
UNDO_KEYS = {ord("u")}
QUIT_KEYS = {27}  # Esc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("unlabeled_dir", help="Directory of unlabeled .jpg crops")
    p.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"Root of labelled-class folders (default: {DATA_DIR})",
    )
    return p.parse_args()


def ensure_data_dirs(data_dir: Path) -> None:
    for cls in CLASSES:
        (data_dir / folder_for(cls)).mkdir(parents=True, exist_ok=True)


def render(crop, info: str):
    h, w = crop.shape[:2]
    view = cv2.resize(
        crop, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    pad = cv2.copyMakeBorder(view, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(
        pad, info, (10, view.shape[0] + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
    )
    cv2.putText(
        pad, "space=. p/n/b/r/q/k (caps=white) s=skip u=undo Esc=quit",
        (10, view.shape[0] + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
    )
    return pad


def unique_dest(folder: Path, src_name: str, prefix: str) -> Path:
    dst = folder / f"{prefix}_{src_name}"
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 2
    while True:
        cand = folder / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main() -> None:
    args = parse_args()
    src = Path(args.unlabeled_dir).resolve()
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    data_dir = Path(args.data_dir).resolve()
    ensure_data_dirs(data_dir)

    crops = sorted(p for p in src.iterdir() if p.suffix.lower() == ".jpg")
    if not crops:
        print(f"no .jpg crops in {src}")
        return

    history: list[tuple[Path, Path]] = []  # (original, current)
    prefix = src.name
    window = "label"
    cv2.namedWindow(window)

    i = 0
    try:
        while i < len(crops):
            crop_path = crops[i]
            img = cv2.imread(str(crop_path))
            if img is None:
                print(f"could not read {crop_path}, skipping")
                i += 1
                continue
            info = f"{i + 1}/{len(crops)}  {crop_path.name}"
            cv2.imshow(window, render(img, info))
            k = cv2.waitKey(0) & 0xFFFF

            if k in QUIT_KEYS:
                break
            if k in SKIP_KEYS:
                i += 1
                continue
            if k in UNDO_KEYS:
                if not history:
                    print("nothing to undo")
                    continue
                orig, curr = history.pop()
                shutil.move(str(curr), str(orig))
                i = crops.index(orig)
                continue
            if k in KEYMAP:
                label = KEYMAP[k]
                folder = data_dir / folder_for(label)
                dst = unique_dest(folder, crop_path.name, prefix)
                shutil.move(str(crop_path), str(dst))
                history.append((crop_path, dst))
                i += 1
                continue
            # unknown key: loop and re-prompt on the same crop
    finally:
        cv2.destroyAllWindows()
        print(f"labelled {len(history)} crops this session")


if __name__ == "__main__":
    main()

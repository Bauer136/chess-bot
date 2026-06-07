"""Render synthetic per-square crops for warm-starting the piece classifier.

Renders random board positions via python-chess's SVG board, rasterizes them
with cairosvg, slices each into 64 labeled crops via `segment_board`, and
writes them into <output>/<class>/*.jpg mirroring the data/ layout so
train.py can read it via `--data-dir`.

This is a warm-start dataset only. SVG glyphs do not look like 3D pieces
under a real camera, so a model trained on synthetic alone will not
generalize. The intended flow:
    1. synthetic.py            -> data_synthetic/
    2. train.py --data-dir data_synthetic --output models/classifier_synth.keras
    3. train.py --data-dir data --init-from models/classifier_synth.keras
       --output models/classifier.keras

Usage:
    python synthetic.py --positions 200
    python synthetic.py --positions 50 --board-size 768 --seed 1
"""

import argparse
import io
import random
from pathlib import Path

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image

from board_state import board_to_labels
from segmentation import iter_square_names, segment_board

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data_synthetic"

# Square-color pairs sampled per board for visual variety. Picked to cover
# a range of light/dark wood-ish tones; the goal is for the model to learn
# that piece identity is invariant to board color, not to match any
# specific tournament set.
LIGHT_PALETTE = ["#ffce9e", "#f0d9b5", "#e8d4ae", "#dfc193", "#eed1a8", "#d8c293"]
DARK_PALETTE = ["#d18b47", "#b58863", "#9c6f3d", "#a67455", "#a07252", "#8a6242"]

# Label -> folder name. `.` (empty) maps to `empty/` because `.` is a
# reserved name on some filesystems.
CLASS_DIR = {
    ".": "empty",
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", type=int, default=200,
                   help="Number of synthetic positions to render. Each yields 64 crops.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--board-size", type=int, default=512,
                   help="Rendered board edge in pixels. Must be divisible by 8.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--jpeg-quality", type=int, default=92,
        help="cv2.imwrite JPEG quality (1-100). Lower = smaller files.",
    )
    return p.parse_args()


def random_position(rng: random.Random) -> chess.Board:
    """Random non-legal piece placements. The classifier only learns per-
    square appearance, so we skip the constraints `Board()` enforces and
    place pieces freely — that gives us, for example, multiple white
    queens or pawns on the back rank, which broadens the training
    distribution at zero cost."""
    board = chess.Board.empty()
    n_pieces = rng.randint(12, 40)
    squares = rng.sample(range(64), n_pieces)
    # Piece pool weighted toward pawns and minor pieces to roughly match
    # the marginal frequency the model will see in real games.
    pool = (
        [chess.PAWN] * 8 + [chess.KNIGHT] * 2 + [chess.BISHOP] * 2
        + [chess.ROOK] * 2 + [chess.QUEEN] * 1 + [chess.KING] * 1
    )
    for sq in squares:
        ptype = rng.choice(pool)
        color = rng.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(sq, chess.Piece(ptype, color))
    return board


def render_board(board: chess.Board, board_size: int, rng: random.Random) -> np.ndarray:
    """Render to a (board_size, board_size, 3) uint8 BGR image."""
    colors = {
        "square light": rng.choice(LIGHT_PALETTE),
        "square dark": rng.choice(DARK_PALETTE),
    }
    svg = chess.svg.board(
        board,
        size=board_size,
        coordinates=False,
        borders=False,
        colors=colors,
    )
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=board_size,
        output_height=board_size,
    )
    rgb = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ensure_class_dirs(output: Path) -> None:
    for folder in CLASS_DIR.values():
        (output / folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if args.board_size % 8 != 0:
        raise SystemExit(f"--board-size must be divisible by 8, got {args.board_size}")
    ensure_class_dirs(output)
    rng = random.Random(args.seed)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    saved = 0
    for i in range(args.positions):
        board = random_position(rng)
        labels = board_to_labels(board)
        img = render_board(board, args.board_size, rng)
        grid = segment_board(img)
        crops = grid.reshape(64, *grid.shape[2:])
        for crop, label, name in zip(crops, labels, iter_square_names()):
            folder = output / CLASS_DIR[label]
            fname = folder / f"synth_{i:04d}_{name}.jpg"
            cv2.imwrite(str(fname), crop, jpeg_params)
            saved += 1
        if (i + 1) % 25 == 0 or i + 1 == args.positions:
            print(f"[synthetic] rendered {i + 1}/{args.positions} positions ({saved} crops)")

    print(f"[synthetic] done: {saved} crops in {output}")


if __name__ == "__main__":
    main()

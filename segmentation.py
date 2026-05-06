"""Stage 4: slice the rectified board into a 64-square grid.

Convention: the rectified frame's top-left corner is square a8. The
calibration's TL click should land on the a8 corner from White's perspective.
If the camera looks at the board from Black's side, that maps Black's TL
click to a8 too — i.e. always click as if you were sitting on White's side
of the rectified output.
"""

from typing import Iterator

import numpy as np

FILES = "abcdefgh"
RANKS = "87654321"  # rank index 0 = rank 8 (top row of the rectified board)


def segment_board(board: np.ndarray) -> np.ndarray:
    """Slice a square board image into an (8, 8, h, w, c) grid.

    grid[0, 0] is a8, grid[7, 7] is h1. For model inference, reshape to
    (64, h, w, c) — that order matches `iter_square_names()`.
    """
    h, w = board.shape[:2]
    if h != w or h % 8 != 0:
        raise ValueError(f"expected NxN board with N divisible by 8, got {h}x{w}")
    s = h // 8
    # (8s, 8s, c) -> (8, s, 8, s, c) -> (8, 8, s, s, c)
    return board.reshape(8, s, 8, s, board.shape[2]).transpose(0, 2, 1, 3, 4)


def square_name(rank_idx: int, file_idx: int) -> str:
    return FILES[file_idx] + RANKS[rank_idx]


def iter_square_names() -> Iterator[str]:
    """Yield names in the same order as grid.reshape(64, ...): a8, b8, ..., h1."""
    for r in range(8):
        for f in range(8):
            yield square_name(r, f)

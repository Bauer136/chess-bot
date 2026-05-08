"""Stage 5: per-square piece classifier.

Path A: this module exposes a stub `predict()` that ignores its input and
returns the standard chess starting position. That keeps Stages 7-10
unblocked while the real CNN is trained on a labeled dataset.

Label convention follows python-chess piece symbols so Stage 7 can feed
them straight into a Board:
    P N B R Q K  -> white
    p n b r q k  -> black
    .            -> empty

Output order matches `segmentation.iter_square_names()` (a8, b8, ..., h1),
which is the order produced by `segment_board(...).reshape(64, ...)`.
"""

from typing import List

import numpy as np

EMPTY = "."

# Starting position in a8 -> h1 order. Eight rows of eight, top rank first.
STARTING_POSITION: List[str] = list(
    "rnbqkbnr"   # rank 8
    "pppppppp"   # rank 7
    + EMPTY * 8  # rank 6
    + EMPTY * 8  # rank 5
    + EMPTY * 8  # rank 4
    + EMPTY * 8  # rank 3
    + "PPPPPPPP" # rank 2
    + "RNBQKBNR" # rank 1
)


def predict(squares_batch: np.ndarray) -> List[str]:
    """Return 64 piece labels for a (64, h, w, c) batch of square crops.

    Stub implementation: ignores pixel content and always returns the
    starting position. Real CNN inference will replace the body without
    changing the call signature.
    """
    if squares_batch.shape[0] != 64:
        raise ValueError(f"expected 64 squares, got {squares_batch.shape[0]}")
    return list(STARTING_POSITION)

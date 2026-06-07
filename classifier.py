"""Stage 5: per-square piece classifier.

If a trained model exists at `models/classifier.keras`, `predict()` runs it.
Otherwise the module falls back to a stub that always returns the standard
starting position, so Stages 7-10 stay testable before the CNN is trained.

Label convention follows python-chess piece symbols so Stage 7 can feed
them straight into a Board:
    P N B R Q K  -> white
    p n b r q k  -> black
    .            -> empty

Output order matches `segmentation.iter_square_names()` (a8, b8, ..., h1),
which is the order produced by `segment_board(...).reshape(64, ...)`.
"""

from pathlib import Path
from typing import List

import numpy as np

EMPTY = "."

# Trained-model output index -> label. Must match train.py CLASS_NAMES order
# with the `empty` folder mapping to `.`.
INDEX_TO_LABEL: List[str] = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

MODEL_PATH = Path(__file__).parent / "models" / "classifier.keras"
INPUT_SIZE = (96, 96)  # must match train.py --input-size at training time

# Starting position in a8 -> h1 order. Returned by the stub fallback.
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

_model = None
_model_load_attempted = False

if not MODEL_PATH.exists():
    print(f"[classifier] {MODEL_PATH} not found — using stub (always returns starting position)")


def _get_model():
    """Lazy-load the Keras model on first inference. Returns None if the
    file is missing — callers fall back to the stub."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf  # deferred so main.py startup stays cheap
    _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict(squares_batch: np.ndarray) -> List[str]:
    """Return 64 piece labels for a (64, h, w, c) uint8 batch in a8 -> h1 order."""
    if squares_batch.shape[0] != 64:
        raise ValueError(f"expected 64 squares, got {squares_batch.shape[0]}")

    model = _get_model()
    if model is None:
        return list(STARTING_POSITION)

    import tensorflow as tf
    # MobileNetV2 preprocess_input lives inside the saved graph, so we only
    # need to land at the right size and float32 here.
    resized = tf.image.resize(squares_batch.astype(np.float32), INPUT_SIZE)
    probs = model.predict(resized, verbose=0)
    return [INDEX_TO_LABEL[i] for i in probs.argmax(axis=1)]

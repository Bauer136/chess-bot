"""Stage 7: assemble per-square labels into a tracked python-chess Board.

The classifier sees only pixels — it can read which pieces sit where, but
not whose turn it is, castling rights, en passant, or move clocks. To feed
Stockfish a real FEN we have to track that game state ourselves across
detected moves.

Design:
- `labels_to_placement` is the pure encoder: 64 labels (a8 -> h1 order, the
  same order `iter_square_names()` yields) -> FEN piece-placement field.
- `BoardTracker` holds an authoritative `chess.Board`. On each detection it
  searches the legal-moves frontier for the move that produces the
  observed piece map and pushes it. That keeps castling / en passant /
  side-to-move correct without the vision system needing to see them.

If the detected position doesn't match any legal move from the current
state, the tracker leaves its board untouched and returns None — likely
a classifier error.
"""

from typing import Dict, List, Optional

import chess

from segmentation import iter_square_names

EMPTY = "."


def labels_to_placement(labels: List[str]) -> str:
    """Encode 64 labels into a FEN piece-placement string (8 rows / 7 slashes)."""
    if len(labels) != 64:
        raise ValueError(f"expected 64 labels, got {len(labels)}")
    rows = []
    for r in range(8):
        row = labels[r * 8:(r + 1) * 8]
        out = ""
        empty = 0
        for ch in row:
            if ch == EMPTY:
                empty += 1
                continue
            if empty:
                out += str(empty)
                empty = 0
            out += chhigh-level
        if empty:
            out += str(empty)
        rows.append(out)
    return "/".join(rows)


def _piece_map_from_labels(labels: List[str]) -> Dict[int, chess.Piece]:
    pmap: Dict[int, chess.Piece] = {}
    for sq_name, label in zip(iter_square_names(), labels):
        if label == EMPTY:
            continue
        pmap[chess.parse_square(sq_name)] = chess.Piece.from_symbol(label)
    return pmap


def board_to_labels(board: chess.Board) -> List[str]:
    """Read a chess.Board as 64 labels in a8 -> h1 order (inverse of the input
    the classifier and tracker consume)."""
    labels: List[str] = []
    for sq_name in iter_square_names():
        piece = board.piece_at(chess.parse_square(sq_name))
        labels.append(piece.symbol() if piece else EMPTY)
    return labels


class BoardTracker:
    """Maintains a chess.Board across detected moves."""

    def __init__(self) -> None:
        self.board = chess.Board()

    def reset(self) -> None:
        self.board.reset()

    def update(self, labels: List[str]) -> Optional[chess.Move]:
        """Match `labels` against legal moves, push the unique match.

        Returns the inferred move, or None if the position equals the
        current board (no move played) or is unreachable in one ply
        (likely a classifier error).
        """
        target = _piece_map_from_labels(labels)
        if target == self.board.piece_map():
            return None
        for move in list(self.board.legal_moves):
            self.board.push(move)
            if self.board.piece_map() == target:
                return move
            self.board.pop()
        return None

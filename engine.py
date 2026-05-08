"""Stage 8: Stockfish UCI bridge.

Wraps `chess.engine.SimpleEngine` so the rest of the pipeline doesn't have
to think about UCI lifecycle. We open Stockfish once at startup and reuse
the same process for every move — popen-per-move would add tens of ms of
startup overhead and waste CPU re-warming the transposition table.
"""

from typing import Optional

import chess
import chess.engine


class EngineWrapper:
    def __init__(self, path: str = "stockfish", think_time: float = 0.5):
        self.path = path
        self.think_time = think_time
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def open(self) -> None:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Return Stockfish's best move from `board`, or None if game is over."""
        if board.is_game_over():
            return None
        if self._engine is None:
            self.open()
        assert self._engine is not None
        result = self._engine.play(board, chess.engine.Limit(time=self.think_time))
        return result.move

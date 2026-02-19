from __future__ import annotations

import logging

import numpy as np

from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


class RowBuffer:
    """Fixed-length circular buffer that reorders completed rows to maintain input order.

    When expression evaluation completes rows out-of-order (due to async processing),
    this buffer collects them and releases them in their original input order.

    How it works:
    - Each row has a position (from row_pos_map) indicating its place in the input sequence
    - Rows are stored at their position offset from the buffer head
    - num_ready tracks how many consecutive rows from the head are available
    - get_rows() only returns rows from this consecutive "ready" region

    Example: If rows arrive in order [2, 0, 1], they're placed at positions 2, 0, 1.
    After row 0 arrives, num_ready=1. After row 1 arrives, num_ready=3 (0,1,2 all present).
    """

    # Maximum capacity of the buffer
    size: int
    # Maps row object id -> position in input sequence; None if order doesn't matter
    row_pos_map: dict[int, int] | None  # id(row) -> position of row in output; None if not maintaining order
    # Total rows currently in buffer (may have gaps if maintaining order)
    num_rows: int  # number of rows in the buffer
    # Consecutive non-None rows from head - these are ready to return
    num_ready: int  # number of consecutive non-None rows at head
    # Circular buffer storage
    buffer: np.ndarray  # of object
    # Current head position within the circular buffer array
    head_idx: int  # index of beginning of the buffer
    # Position in input sequence that corresponds to head_idx
    head_pos: int  # row position of the beginning of the buffer

    def __init__(self, size: int):
        self.size = size
        self.row_pos_map = None
        self.num_rows = 0
        self.num_ready = 0
        self.buffer = np.full(size, None, dtype=object)
        self.head_pos = 0
        self.head_idx = 0

    def set_row_pos_map(self, row_pos_map: dict[int, int]) -> None:
        """Enable order-preserving mode by providing a position map."""
        self.row_pos_map = row_pos_map

    def add_row(self, row: exprs.DataRow) -> None:
        """Add a completed row to the buffer at its correct position."""
        # Calculate where in the buffer this row belongs
        offset: int  # of new row from head
        if self.row_pos_map is not None:
            # Order-preserving mode: place at position determined by input order
            pos = self.row_pos_map.get(id(row))
            assert pos is not None and (pos - self.head_pos < self.size), f'{pos} {self.head_pos} {self.size}'
            offset = pos - self.head_pos
        else:
            # FIFO mode: append at next available slot
            offset = self.num_rows
        idx = (self.head_idx + offset) % self.size
        assert self.buffer[idx] is None

        self.buffer[idx] = row
        self.num_rows += 1
        if self.row_pos_map is not None:
            # Check if this row extends the consecutive "ready" region from the head
            if offset == self.num_ready:
                # Scan forward to find the new num_ready (consecutive non-None slots)
                while offset < self.size and self.buffer[(self.head_idx + offset) % self.size] is not None:
                    offset += 1
                self.num_ready = offset
        else:
            self.num_ready += 1

    def get_rows(self, n: int) -> list[exprs.DataRow]:
        """Remove and return up to n consecutive ready rows from the head.

        Only returns rows from the "ready" region (consecutive non-None from head).
        Handles circular buffer wraparound.
        """
        n = min(n, self.num_ready)
        if n == 0:
            return []
        rows: list[exprs.DataRow]
        # Handle potential wraparound in circular buffer
        if self.head_idx + n <= self.size:
            rows = self.buffer[self.head_idx : self.head_idx + n].tolist()
            self.buffer[self.head_idx : self.head_idx + n] = None
        else:
            rows = np.concatenate([self.buffer[self.head_idx :], self.buffer[: self.head_idx + n - self.size]]).tolist()
            self.buffer[self.head_idx :] = None
            self.buffer[: self.head_idx + n - self.size] = None
        self.head_pos += n
        self.head_idx = (self.head_idx + n) % self.size
        self.num_rows -= n
        self.num_ready -= n
        return rows

from __future__ import annotations

from typing import Iterator

from pixeltable import exprs


class DataRowBatch:
    """A collection of DataRows that flow through the execution pipeline together.

    DataRowBatch is the unit of data transfer between execution nodes (ExecNode subclasses).
    It bundles rows with their associated RowBuilder, which contains the expression DAG
    metadata needed to interpret slot indices.

    The batch is used to:
    - Transfer rows between pipeline stages (e.g., from scan to expression evaluation)
    - Group rows for efficient batched operations (e.g., batched function calls)
    - Maintain row ordering when needed

    Typical lifecycle:
    1. Created by a source node (e.g., SqlScanNode) with rows from database
    2. Passed to ExprEvalNode for computed column evaluation
    3. Yielded to the caller or passed to next pipeline stage
    """

    # Provides expression DAG metadata and factory methods for creating compatible DataRows
    row_builder: exprs.RowBuilder
    # The actual data rows in this batch
    rows: list[exprs.DataRow]

    def __init__(self, row_builder: exprs.RowBuilder, rows: list[exprs.DataRow] | None = None):
        """Create a batch with the given RowBuilder configuration.

        Args:
            row_builder: Defines the expression slots and types for rows in this batch.
            rows: Optional pre-existing rows. If None, starts with an empty list.
        """
        self.row_builder = row_builder
        self.rows = [] if rows is None else rows

    def add_row(self, row: exprs.DataRow | None) -> exprs.DataRow:
        """Add a row to the batch, creating a new one if needed.

        Args:
            row: Existing DataRow to add, or None to create a new one from row_builder.

        Returns:
            The added DataRow (newly created if row was None).
        """
        if row is None:
            row = self.row_builder.make_row()
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        """Remove and return the last row from the batch."""
        return self.rows.pop()

    def __len__(self) -> int:
        """Return the number of rows in this batch."""
        return len(self.rows)

    def __getitem__(self, index: int) -> exprs.DataRow:
        """Get a row by index."""
        return self.rows[index]

    def __iter__(self) -> Iterator[exprs.DataRow]:
        """Iterate over rows in the batch."""
        return iter(self.rows)

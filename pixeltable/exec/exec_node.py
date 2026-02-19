from __future__ import annotations

import abc
import logging
from types import TracebackType
from typing import AsyncIterator, Iterable, Iterator, TypeVar

from typing_extensions import Self

from pixeltable import exprs
from pixeltable.env import Env

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext

_logger = logging.getLogger('pixeltable')


class ExecNode(abc.ABC):
    """
    Base class of all execution nodes in the query execution pipeline.

    ExecNode forms a tree structure where each node:
    - Receives DataRowBatch objects from its input node (if any)
    - Processes or transforms the rows
    - Yields DataRowBatch objects to its parent node

    The pipeline is executed lazily via async iteration (__aiter__). Nodes include:
    - SqlScanNode: Reads rows from database
    - ExprEvalNode: Evaluates computed columns and expressions
    - CachePrefetchNode: Pre-fetches media files from remote storage
    - AggregationNode: Computes aggregates (sum, count, etc.)

    Lifecycle: set_ctx() -> __enter__() -> __aiter__() -> __exit__()
    """

    # Expressions this node needs to produce in its output
    output_exprs: Iterable[exprs.Expr]
    # Shared expression DAG metadata and DataRow factory
    row_builder: exprs.RowBuilder
    # Upstream node that feeds rows into this node (None for source nodes)
    input: ExecNode | None
    # Image slots that need to be flushed after use (not part of final output, but needed for computation)
    flushed_img_slots: list[int]  # idxs of image slots of our output_exprs dependencies
    # Execution context with progress reporting and error handling config
    ctx: ExecContext | None

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode | None = None,
    ):
        assert all(expr.is_valid for expr in output_exprs)
        self.output_exprs = output_exprs
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.ctx = input.ctx if input is not None else None

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        pass

    def __iter__(self) -> Iterator[DataRowBatch]:
        """Synchronous iteration wrapper that runs the async iterator on the event loop.

        Allows ExecNode to be used in regular for loops when async context isn't available.
        """
        loop = Env.get().event_loop
        aiter = self.__aiter__()
        try:
            while True:
                batch: DataRowBatch = loop.run_until_complete(aiter.__anext__())
                yield batch
        except StopAsyncIteration:
            pass
        # TODO:
        #  - we seem to have some tasks that aren't accounted for by ExprEvalNode and don't get cancelled by the time
        #    we end up here
        # - however, blindly cancelling all pending tasks doesn't work when running in a jupyter environment, which
        #   creates tasks on its own

    def __enter__(self) -> Self:
        if self.ctx.show_progress:
            self.ctx.start_progress()
        self._open_aux()
        return self

    def _open_aux(self) -> None:
        """Call _open() bottom-up"""
        if self.input is not None:
            self.input._open_aux()
        self._open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        # Ensure progress stops on exit (including empty results, errors, interrupts)
        Env.get().stop_progress()
        self._close_aux()

    def _close_aux(self) -> None:
        """Call _close() top-down"""
        self._close()
        if self.input is not None:
            self.input._close_aux()

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    T = TypeVar('T', bound='ExecNode')

    def get_node(self, node_class: type[T]) -> T | None:
        """Find the first node of a specific type in the pipeline, searching from this node downward."""
        if isinstance(self, node_class):
            return self
        if self.input is not None:
            return self.input.get_node(node_class)
        return None

    def set_limit(self, limit: int) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_limit(limit)

    def set_offset(self, offset: int) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_offset(offset)

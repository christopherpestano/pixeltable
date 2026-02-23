from __future__ import annotations

import asyncio
import logging
import traceback
from types import TracebackType
from typing import AsyncIterator, Iterable

import numpy as np

import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.utils.progress_reporter import ProgressReporter

from ..data_row_batch import DataRowBatch
from ..exec_node import ExecNode
from .evaluators import FnCallEvaluator, NestedRowList
from .globals import ExprEvalCtx, Scheduler
from .row_buffer import RowBuffer
from .schedulers import SCHEDULERS

_logger = logging.getLogger('pixeltable')


class ExprEvalNode(ExecNode):
    """
    Expression evaluation

    Resource management:
    - the execution system tries to limit total memory consumption by limiting the number of rows that are in
      circulation
    - during execution, slots that aren't part of the output are garbage collected as soon as their direct dependents
      are materialized

    TODO:
    - Literal handling: currently, Literal values are copied into slots via the normal evaluation mechanism, which is
      needless overhead; instead: pre-populate Literal slots in _init_row()
    - dynamically determine MAX_BUFFERED_ROWS, based on the avg memory consumption of a row and our configured memory
      limit
    - local model inference on gpu: currently, no attempt is made to ensure that models can fit onto the gpu
      simultaneously, which will cause errors; instead, the execution should be divided into sequential phases, each
      of which only contains a subset of the models which is known to fit onto the gpu simultaneously
    """

    # Controls whether output order matches input order (True) or allows out-of-order completion (False)
    maintain_input_order: bool  # True if we're returning rows in the order we received them from our input
    outputs: np.ndarray  # bool per slot; True if this slot is part of our output
    schedulers: dict[str, Scheduler]  # key: resource pool name
    eval_ctx: ExprEvalCtx  # for input/output rows

    # execution state
    # Holds references to running tasks to prevent garbage collection before completion
    tasks: set[asyncio.Task]  # collects all running tasks to prevent them from getting gc'd
    exc_event: asyncio.Event  # set if an exception needs to be propagated
    error: Exception | None  # exception that needs to be propagated
    completed_rows: asyncio.Queue[exprs.DataRow]  # rows that have completed evaluation
    completed_event: asyncio.Event  # set when completed_rows is non-empty
    input_iter: AsyncIterator[DataRowBatch]
    current_input_batch: DataRowBatch | None  # batch from which we're currently consuming rows
    input_row_idx: int  # next row to consume from current_input_batch
    next_input_batch: DataRowBatch | None  # read-ahead input batch
    available_input_rows: int  # total number across both current_/next_input_batch
    input_complete: bool  # True if we've received all input batches
    # Counts rows that have been dispatched for evaluation but not yet completed
    num_in_flight: int  # number of dispatched rows that haven't completed
    row_pos_map: dict[int, int] | None  # id(row) -> position of row in input; only set if maintain_input_order
    output_buffer: RowBuffer  # holds rows that are ready to be returned, in order
    progress_reporter: ProgressReporter | None

    # debugging
    num_input_rows: int
    num_output_rows: int

    BATCH_SIZE = 64
    # Limits memory usage by capping rows in flight (dispatched but not yet returned to caller)
    MAX_BUFFERED_ROWS = 2048  # maximum number of rows that have been dispatched but not yet returned

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode,
        maintain_input_order: bool = True,
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.maintain_input_order = maintain_input_order
        self.outputs = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.outputs[output_slot_idxs] = True
        self.tasks = set()
        self.error = None

        self.input_iter = self.input.__aiter__()
        self.current_input_batch = None
        self.next_input_batch = None
        self.input_row_idx = 0
        self.available_input_rows = 0
        self.input_complete = False
        self.num_in_flight = 0
        self.row_pos_map = None
        self.output_buffer = RowBuffer(self.MAX_BUFFERED_ROWS)
        self.progress_reporter = None

        self.num_input_rows = 0
        self.num_output_rows = 0

        # self.slot_evaluators = {}
        self.schedulers = {}
        # self._init_slot_evaluators()
        self.eval_ctx = ExprEvalCtx(self, self.row_builder, output_exprs, input_exprs)

    def _open(self) -> None:
        self.progress_reporter = self.ctx.add_progress_reporter('Cell computations', 'cells')

    def set_input_order(self, maintain_input_order: bool) -> None:
        self.maintain_input_order = maintain_input_order

    async def _fetch_input_batch(self) -> None:
        """
        Fetches another batch from our input or sets input_complete to True if there are no more batches.

        - stores the batch in current_input_batch, if not already set, or next_input_batch
        - updates row_pos_map, if needed
        """
        assert not self.input_complete
        try:
            batch = await anext(self.input_iter)
            if self.progress_reporter is not None:
                # make sure our progress reporter shows up before we run anything long
                self.progress_reporter.update(0)
            assert self.next_input_batch is None
            if self.current_input_batch is None:
                self.current_input_batch = batch
            else:
                self.next_input_batch = batch
            if self.maintain_input_order:
                for idx, row in enumerate(batch.rows):
                    self.row_pos_map[id(row)] = self.num_input_rows + idx
            self.num_input_rows += len(batch)
            self.available_input_rows += len(batch)
            _logger.debug(
                f'adding input: batch_size={len(batch)} #input_rows={self.num_input_rows} '
                f'#avail={self.available_input_rows}'
            )
        except StopAsyncIteration:
            self.input_complete = True
            _logger.debug(f'finished input: #input_rows={self.num_input_rows}, #avail={self.available_input_rows}')
        # make sure to pass DBAPIError through, so the transaction handling logic sees it
        except Exception as exc:
            self.error = exc
            self.exc_event.set()

    @property
    def total_buffered(self) -> int:
        return self.num_in_flight + self.completed_rows.qsize() + self.output_buffer.num_rows

    def _dispatch_input_rows(self) -> None:
        """Dispatch the maximum number of input rows, given total_buffered; does not block"""
        if self.available_input_rows == 0:
            return
        num_rows = min(self.MAX_BUFFERED_ROWS - self.total_buffered, self.available_input_rows)
        assert num_rows >= 0
        if num_rows == 0:
            return
        assert self.current_input_batch is not None
        available_in_current_batch = len(self.current_input_batch) - self.input_row_idx

        rows: list[exprs.DataRow]
        if available_in_current_batch > num_rows:
            # we only need rows from current_input_batch
            rows = self.current_input_batch.rows[self.input_row_idx : self.input_row_idx + num_rows]
            self.input_row_idx += num_rows
        else:
            # we need rows from both current_/next_input_batch
            rows = self.current_input_batch.rows[self.input_row_idx :]
            self.current_input_batch = self.next_input_batch
            self.next_input_batch = None
            self.input_row_idx = 0
            num_remaining = num_rows - len(rows)
            if num_remaining > 0:
                rows.extend(self.current_input_batch.rows[:num_remaining])
                self.input_row_idx = num_remaining
        self.available_input_rows -= num_rows
        self.num_in_flight += num_rows
        self._log_state(f'dispatch input ({num_rows})')

        self.eval_ctx.init_rows(rows)
        self.dispatch(rows, self.eval_ctx)

    def _log_state(self, prefix: str) -> None:
        _logger.debug(
            f'{prefix}: #in-flight={self.num_in_flight} #complete={self.completed_rows.qsize()} '
            f'#output-buffer={self.output_buffer.num_rows} #ready={self.output_buffer.num_ready} '
            f'total-buffered={self.total_buffered} #avail={self.available_input_rows} '
            f'#input={self.num_input_rows} #output={self.num_output_rows}'
        )

    def _init_schedulers(self) -> None:
        resource_pools = {
            eval.fn_call.resource_pool
            for eval in self.eval_ctx.slot_evaluators.values()
            if isinstance(eval, FnCallEvaluator)
        }
        resource_pools = {pool for pool in resource_pools if pool is not None}
        for pool_name in resource_pools:
            for scheduler in SCHEDULERS:
                if scheduler.matches(pool_name):
                    self.schedulers[pool_name] = scheduler(pool_name, self)
                    break
            if pool_name not in self.schedulers:
                raise RuntimeError(f'No scheduler found for resource pool {pool_name}')

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        """
        Main event loop

        Goals:
        - return completed DataRowBatches as soon as they become available
        - maximize the number of rows in flight in order to maximize parallelism, up to the given limit
        """
        # initialize completed_rows and events, now that we have the correct event loop
        self.completed_rows = asyncio.Queue[exprs.DataRow]()
        self.exc_event = asyncio.Event()
        self.completed_event = asyncio.Event()
        self._init_schedulers()
        if self.maintain_input_order:
            self.row_pos_map = {}
            self.output_buffer.set_row_pos_map(self.row_pos_map)

        row: exprs.DataRow
        exc_wait_task = asyncio.create_task(self.exc_event.wait(), name='exc_event.wait()')
        fetch_input_task: asyncio.Task | None = None
        completion_wait_task: asyncio.Task | None = None
        closed_evaluators = False  # True after calling Evaluator.close()
        exprs.Expr.prepare_list(self.eval_ctx.all_exprs)

        # === MAIN EVENT LOOP - processes rows through expression evaluation pipeline ===
        try:
            while True:
                # STEP 1: Process completed rows - move them from completion queue to output buffer
                while not self.completed_rows.empty():
                    # move completed rows to output buffer
                    while not self.completed_rows.empty():
                        row = self.completed_rows.get_nowait()
                        self.output_buffer.add_row(row)
                        if self.row_pos_map is not None:
                            self.row_pos_map.pop(id(row))

                    self._log_state('processed completed')
                    # return as many batches as we have available
                    while self.output_buffer.num_ready >= self.BATCH_SIZE:
                        batch_rows = self.output_buffer.get_rows(self.BATCH_SIZE)
                        self.num_output_rows += len(batch_rows)
                        # make sure we top up our in-flight rows before yielding
                        self._dispatch_input_rows()
                        self._log_state(f'yielding {len(batch_rows)} rows')
                        yield DataRowBatch(row_builder=self.row_builder, rows=batch_rows)
                        # at this point, we may have more completed rows

                # STEP 2: Check termination conditions
                assert self.completed_rows.empty()  # all completed rows should be sitting in output_buffer
                self.completed_event.clear()
                if self.input_complete and self.num_in_flight == 0:
                    # there is no more input and nothing left to wait for
                    assert self.available_input_rows == 0
                    if self.output_buffer.num_ready > 0:
                        assert self.output_buffer.num_rows == self.output_buffer.num_ready
                        # yield the leftover rows
                        batch_rows = self.output_buffer.get_rows(self.output_buffer.num_ready)
                        self.num_output_rows += len(batch_rows)
                        self._log_state(f'yielding {len(batch_rows)} rows')
                        yield DataRowBatch(row_builder=self.row_builder, rows=batch_rows)

                    assert self.output_buffer.num_rows == 0
                    return

                # STEP 3: Close evaluators if no more input (flush any queued batches)
                if self.input_complete and self.available_input_rows == 0 and not closed_evaluators:
                    # no more input rows to dispatch, but we're still waiting for rows to finish:
                    # close all slot evaluators to flush queued rows
                    for evaluator in self.eval_ctx.slot_evaluators.values():
                        evaluator.close()
                    closed_evaluators = True

                # STEP 4: Wait for next event (input batch, completion, or error)
                # we don't have a full batch of rows at this point and need to wait
                pending_tasks = {exc_wait_task}  # always wait for an exception
                if self.next_input_batch is None and not self.input_complete:
                    # also wait for another batch if we don't have a read-ahead batch yet
                    if fetch_input_task is None:
                        fetch_input_task = asyncio.create_task(self._fetch_input_batch(), name='_fetch_input_batch()')
                    pending_tasks.add(fetch_input_task)
                if self.num_in_flight > 0:
                    # also wait for more rows to complete
                    if completion_wait_task is None:
                        completion_wait_task = asyncio.create_task(self.completed_event.wait(), name='completed.wait()')
                    pending_tasks.add(completion_wait_task)
                done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

                if self.exc_event.is_set():
                    # we got an exception that we need to propagate through __iter__()
                    if isinstance(self.error, excs.ExprEvalError):
                        raise self.error from self.error.exc
                    else:
                        raise self.error
                if completion_wait_task in done:
                    self._log_state('completion_wait_task done')
                    completion_wait_task = None
                if fetch_input_task in done:
                    self._dispatch_input_rows()
                    fetch_input_task = None

        finally:
            # task cleanup
            active_tasks = {exc_wait_task}
            if fetch_input_task is not None:
                active_tasks.add(fetch_input_task)
            if completion_wait_task is not None:
                active_tasks.add(completion_wait_task)
            active_tasks.update(self.tasks)
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            _ = await asyncio.gather(*active_tasks, return_exceptions=True)

            # expr cleanup
            exprs.Expr.release_list(self.eval_ctx.all_exprs)

    def dispatch_exc(
        self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType, exec_ctx: ExprEvalCtx
    ) -> None:
        """Propagate exception to main event loop or to dependent slots, depending on ignore_errors"""
        if len(rows) == 0 or self.exc_event.is_set():
            return

        if not self.ctx.ignore_errors:
            dependency_idxs = [e.slot_idx for e in exec_ctx.row_builder.unique_exprs[slot_with_exc].dependencies()]
            first_row = rows[0]
            input_vals = [first_row[idx] for idx in dependency_idxs]
            e = exec_ctx.row_builder.unique_exprs[slot_with_exc]
            self.error = excs.ExprEvalError(e, f'expression {e}', first_row.get_exc(e.slot_idx), exc_tb, input_vals, 0)
            self.exc_event.set()
            return

        for row in rows:
            assert row.has_exc(slot_with_exc)
            exc = row.get_exc(slot_with_exc)
            # propagate exception
            for slot_idx in np.nonzero(exec_ctx.row_builder.transitive_dependents[slot_with_exc])[0].tolist():
                row.set_exc(slot_idx, exc)
        self.dispatch(rows, exec_ctx)

    def dispatch(self, rows: list[exprs.DataRow], exec_ctx: ExprEvalCtx) -> None:
        """
        Dispatch rows to slot evaluators, based on materialized dependencies.

        This is the CORE SCHEDULING FUNCTION of the expression evaluation engine.
        It determines which "slots" (think: columns or computed expressions) are ready
        to be evaluated for each row, and schedules them for execution.

        CONCEPTUAL OVERVIEW:
        --------------------
        Imagine a spreadsheet where some cells depend on other cells (like formulas).
        You can't compute cell C until cells A and B have values (if C = A + B).
        This method figures out which cells are "ready" (all their inputs are available)
        and sends them off to be computed.

        TERMINOLOGY:
        ------------
        - "Slot": A position in a row that holds a value. Each expression/column gets a slot.
                  Think of it as a cell in a spreadsheet row.
        - "Row": A single data record with multiple slots (like a row in a spreadsheet).
        - "Dependencies": Other slots that must be computed BEFORE a given slot can be computed.
                          For example, if slot 3 = slot 1 + slot 2, then slots 1 and 2 are
                          dependencies of slot 3.
        - "Materialized": A slot is materialized when it has an actual value computed and stored.
        - "Evaluator": An object that knows how to compute the value for a specific slot.

        WHAT THIS METHOD DOES (step by step):
        -------------------------------------
        1. For each row, figure out which slots are now "ready" to be evaluated
        2. Track which rows are completely done (all output slots have values)
        3. Clean up memory by garbage-collecting intermediate values we no longer need
        4. Send completed rows to the output queue
        5. Schedule ready slots to their evaluators for actual computation

        Args:
            rows: List of DataRow objects to process. Each row contains slots that may
                  or may not have values yet.
            exec_ctx: The ExprEvalCtx (expression evaluation context) that holds metadata
                      about how expressions relate to each other (dependencies, evaluators, etc.)
        """
        # =====================================================================
        # EARLY EXIT CHECK
        # =====================================================================
        # If there are no rows to process, or if an exception has already been raised
        # somewhere else in the system (exc_event is set), just return immediately.
        # No point doing work if we're going to throw everything away due to an error.
        if len(rows) == 0 or self.exc_event.is_set():
            return

        # =====================================================================
        # INITIALIZE TRACKING MATRICES
        # =====================================================================
        # We use NumPy boolean arrays for efficient batch operations.

        # ready_slots: A 2D matrix where ready_slots[row_index][slot_index] = True
        # means "for this row, this slot is ready to be evaluated (all its dependencies
        # have values)".
        # Shape: (number of rows) x (number of slots in a row)
        # Example: If we have 3 rows and 5 slots, this is a 3x5 matrix of False values initially.
        ready_slots = np.zeros((len(rows), exec_ctx.row_builder.num_materialized), dtype=bool)

        # completed_rows: A 1D array where completed_rows[row_index] = True means
        # "this row is completely done - all its output slots have values".
        # Shape: (number of rows,)
        completed_rows = np.zeros(len(rows), dtype=bool)

        # num_computed_outputs: Counter for how many output slot values we computed.
        # Used for progress reporting (showing the user how much work has been done).
        num_computed_outputs = 0

        # =====================================================================
        # MAIN LOOP: Process each row to determine what's ready and what's done
        # =====================================================================
        for i, row in enumerate(rows):
            # -----------------------------------------------------------------
            # SHAPE VALIDATION (currently a no-op, but left for debugging)
            # -----------------------------------------------------------------
            # This check verifies that the row's missing_slots array has the same
            # shape as self.outputs. If they don't match, something is wrong with
            # how the row was constructed. The 'pass' here suggests this might be
            # a placeholder for future error handling or debugging code.
            if row.missing_slots.shape != self.outputs.shape:
                pass

            # -----------------------------------------------------------------
            # TRACK PROGRESS FOR OUTPUT SLOTS
            # -----------------------------------------------------------------
            # We only count newly-computed OUTPUT slots (the final results the user wants),
            # not intermediate computed values. This is for the progress bar.
            #
            # self.eval_ctx is the "top-level" context for the main query.
            # exec_ctx might be a nested context (e.g., inside a JSON mapper).
            # We only report progress for the top-level context.
            if self.eval_ctx is exec_ctx:
                # Count how many output slots are STILL missing (before we update).
                # row.missing_slots: boolean array, True = "this slot still needs a value"
                # self.outputs: boolean array, True = "this slot is an output we care about"
                # The bitwise AND gives us "output slots that are still missing".
                missing_outputs = (row.missing_slots & self.outputs).sum()

                # Update missing_slots: a slot is still "missing" only if it BOTH:
                #   1. Was marked as missing (row.missing_slots is True)
                #   2. Still doesn't have a value (row.has_val is False)
                # After this line, missing_slots only contains slots that are TRULY still missing.
                row.missing_slots &= row.has_val == False

                # Count how many output slots we just "computed" (were missing before, have values now).
                # This is: (missing before) - (missing after)
                num_computed_outputs += missing_outputs - (row.missing_slots & self.outputs).sum()
            else:
                # For nested contexts, we still update missing_slots but don't track progress.
                row.missing_slots &= row.has_val == False

            # -----------------------------------------------------------------
            # CHECK IF ROW IS COMPLETE
            # -----------------------------------------------------------------
            # If no slots are missing (sum of missing_slots is 0), the row is done!
            if row.missing_slots.sum() == 0:
                completed_rows[i] = True
            else:
                # -----------------------------------------------------------------
                # DETERMINE WHICH SLOTS ARE READY FOR EVALUATION
                # -----------------------------------------------------------------
                # A slot is "ready" when ALL of the following are true:
                #   1. All its dependencies have values (no missing dependencies)
                #   2. It hasn't been scheduled for evaluation yet
                #   3. It's still missing (we haven't computed it yet)

                # Step A: Get the number of dependencies for each MISSING slot.
                #
                # EXAMPLE: Suppose we have 5 slots representing these expressions:
                #   Slot 0: input column "name"        (no dependencies)
                #   Slot 1: input column "age"         (no dependencies)
                #   Slot 2: upper(name)                (depends on slot 0)
                #   Slot 3: age + 10                   (depends on slot 1)
                #   Slot 4: concat(upper(name), age)   (depends on slots 2 and 3)
                #
                # num_dependencies would be: [0, 0, 1, 1, 2]
                #   - Slot 0 has 0 dependencies (it's an input)
                #   - Slot 1 has 0 dependencies (it's an input)
                #   - Slot 2 has 1 dependency (slot 0)
                #   - Slot 3 has 1 dependency (slot 1)
                #   - Slot 4 has 2 dependencies (slots 2 and 3)
                #
                # If missing_slots is [False, False, True, True, True] (we need slots 2,3,4):
                # missing_dependencies = [0, 0, 1, 1, 2] * [False, False, True, True, True]
                #                      = [0, 0, 1, 1, 2]  (zeros out slots we don't care about)
                missing_dependencies = exec_ctx.row_builder.num_dependencies * row.missing_slots

                # Step B: Count how many dependencies are already MATERIALIZED (have values).
                #
                # dependencies is a 2D matrix (slots x slots) where dependencies[i][j] = True
                # means "slot i depends on slot j".
                #
                # EXAMPLE (continuing from above):
                # dependencies matrix:
                #        slot0  slot1  slot2  slot3  slot4
                # slot0 [False, False, False, False, False]  (no deps)
                # slot1 [False, False, False, False, False]  (no deps)
                # slot2 [True,  False, False, False, False]  (depends on slot 0)
                # slot3 [False, True,  False, False, False]  (depends on slot 1)
                # slot4 [False, False, True,  True,  False]  (depends on slots 2 and 3)
                #
                # If has_val is [True, True, False, False, False] (inputs are populated):
                # dependencies * has_val broadcasts has_val across rows:
                #        [True, True, False, False, False]
                # slot2: [True, False, False, False, False] → sum = 1 (slot 0 is ready)
                # slot3: [False, True, False, False, False] → sum = 1 (slot 1 is ready)
                # slot4: [False, False, False, False, False] → sum = 0 (slots 2,3 not ready)
                #
                # So num_mat_dependencies = [0, 0, 1, 1, 0]
                num_mat_dependencies = np.sum(exec_ctx.row_builder.dependencies * row.has_val, axis=1)

                # Step C: Calculate how many dependencies are still MISSING for each slot.
                #
                # EXAMPLE (continuing):
                # missing_dependencies = [0, 0, 1, 1, 2]
                # num_mat_dependencies = [0, 0, 1, 1, 0]
                # num_missing           = [0, 0, 0, 0, 2]
                #
                # Slots 2 and 3 have num_missing=0, meaning ALL their dependencies are satisfied!
                # Slot 4 has num_missing=2, meaning it's still waiting for 2 dependencies.
                num_missing = missing_dependencies - num_mat_dependencies

                # Step D: A slot is "ready" if:
                #   - num_missing == 0 (all dependencies satisfied)
                #   - is_scheduled == False (not already queued for evaluation)
                #   - missing_slots == True (we still need to compute it)
                #
                # EXAMPLE (continuing):
                # num_missing == 0:    [True,  True,  True,  True,  False]
                # is_scheduled==False: [True,  True,  True,  True,  True ]  (nothing scheduled yet)
                # missing_slots:       [False, False, True,  True,  True ]
                # Result (AND all):    [False, False, True,  True,  False]
                #
                # So slots 2 and 3 are ready! Slot 4 is not ready (still waiting on 2 and 3).
                ready_slots[i] = (num_missing == 0) & (row.is_scheduled == False) & row.missing_slots

                # Step E: Mark these slots as scheduled so we don't schedule them again.
                row.is_scheduled |= ready_slots[i]

            # -----------------------------------------------------------------
            # GARBAGE COLLECTION: Free memory from intermediate values
            # -----------------------------------------------------------------
            # Once all slots that depend on an intermediate value have been computed,
            # we can clear that intermediate value to save memory.
            #
            # Think of it like this: if you computed A and B to get C, and C was the
            # only thing that needed A and B, you can throw away A and B now.

            # Count how many NOT-YET-COMPUTED slots depend on each slot.
            # dependencies[row.has_val == False] selects rows of the dependency matrix
            # for slots that DON'T have values yet.
            # Summing along axis=0 counts, for each slot, how many uncomputed slots depend on it.
            missing_dependents = np.sum(exec_ctx.row_builder.dependencies[row.has_val == False], axis=0)

            # A slot is a garbage collection target if:
            #   1. missing_dependents == 0: No uncomputed slots need this value anymore
            #   2. row.missing_dependents > 0: It USED to have dependents (meaning it was
            #      an intermediate value, not just an unused slot)
            #   3. exec_ctx.gc_targets: It's marked as OK to garbage collect (not an output)
            gc_targets = (missing_dependents == 0) & (row.missing_dependents > 0) & exec_ctx.gc_targets

            # Actually clear the values in the row for these slots.
            row.clear(gc_targets)

            # Update the row's tracking of how many dependents each slot has.
            row.missing_dependents = missing_dependents

        # =====================================================================
        # REPORT PROGRESS
        # =====================================================================
        # If we have a progress reporter and we computed some output values,
        # tell the progress bar to update.
        if self.progress_reporter is not None and num_computed_outputs > 0:
            self.progress_reporter.update(int(num_computed_outputs))

        # =====================================================================
        # HANDLE COMPLETED ROWS
        # =====================================================================
        # Some rows may now be fully complete (all output slots have values).
        # We need to either:
        #   A) If they're "nested rows" (sub-rows of a parent row), notify the parent.
        #   B) If they're top-level rows, put them in the output queue.
        if np.any(completed_rows):
            # Get the indices of all completed rows (where completed_rows[i] == True).
            # .nonzero() returns a tuple of arrays; [0] gets the first (and only) array.
            completed_idxs = list(completed_rows.nonzero()[0])

            # Check if these are nested rows (rows that belong to a parent row).
            # Nested rows are used for things like JSON array expansion, where one
            # parent row produces multiple child rows.
            if rows[i].parent_row is not None:
                # NESTED ROWS: Notify the parent that child rows are complete.
                for i in completed_idxs:
                    row = rows[i]
                    # Verify this is actually a nested row with proper parent linkage.
                    assert row.parent_row is not None and row.parent_slot_idx is not None
                    # The parent's slot contains a NestedRowList that tracks all child rows.
                    assert isinstance(row.parent_row.vals[row.parent_slot_idx], NestedRowList)
                    # Tell the NestedRowList that one more row is complete.
                    row.parent_row.vals[row.parent_slot_idx].complete_row()
            else:
                # TOP-LEVEL ROWS: Put them in the output queue for the main event loop.
                for i in completed_idxs:
                    # put_nowait: Add to queue without blocking (we know queue won't be full).
                    self.completed_rows.put_nowait(rows[i])
                # Signal that there are completed rows ready to be consumed.
                self.completed_event.set()
                # Decrease the count of rows we're still processing.
                self.num_in_flight -= len(completed_idxs)

        # =====================================================================
        # SCHEDULE READY SLOTS FOR EVALUATION
        # =====================================================================
        # Now we actually send the ready slots to their evaluators to be computed.
        #
        # np.sum(ready_slots, axis=0) sums along the rows, giving us a count per slot
        # of how many rows have that slot ready.
        # .nonzero()[0] gives us the indices of slots where at least one row is ready.
        for slot_idx in np.sum(ready_slots, axis=0).nonzero()[0]:
            # Get the column of ready_slots for this slot (which rows have it ready?).
            ready_rows_v = ready_slots[:, slot_idx].flatten()

            # Find the indices of rows where this slot is ready.
            _ = ready_rows_v.nonzero()  # This line appears to be dead code / debugging artifact

            # Build a list of the actual row objects that are ready for this slot.
            ready_rows = [rows[i] for i in ready_rows_v.nonzero()[0]]

            # Log for debugging: how many rows are being scheduled for this slot.
            _logger.debug(f'Scheduling {len(ready_rows)} rows for slot {slot_idx}')

            # Actually schedule the rows with the slot's evaluator.
            # The evaluator knows how to compute values for this slot (e.g., run a function,
            # call an API, etc.). It will eventually produce values and call dispatch()
            # again when those values are ready.
            exec_ctx.slot_evaluators[slot_idx].schedule(ready_rows, slot_idx)

    def register_task(self, t: asyncio.Task) -> None:
        self.tasks.add(t)
        t.add_done_callback(self._done_cb)

    def _done_cb(self, t: asyncio.Task) -> None:
        self.tasks.discard(t)
        # end the main loop if we had an unhandled exception
        try:
            t.result()
        except KeyboardInterrupt:
            # ExprEvalNode instances are long-running and reused across multiple operations.
            # When a user interrupts an operation (Ctrl+C), the main evaluation loop properly
            # handles the KeyboardInterrupt and terminates the current operation. However,
            # background tasks spawned by evaluators may complete asynchronously after the
            # operation has ended, and their done callbacks will fire during subsequent
            # operations. These "phantom" KeyboardInterrupt exceptions from previous
            # operations' background tasks should not interfere with new operations, so we
            # absorb them here rather than propagating them via self.error/self.exc_event.
            _logger.debug('Task completed with KeyboardInterrupt (user cancellation)')
            pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            stack_trace = traceback.format_exc()
            self.error = excs.Error(f'Exception in task: {exc}\n{stack_trace}')
            self.exc_event.set()

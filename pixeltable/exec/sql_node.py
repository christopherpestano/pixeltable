import datetime
import logging
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING, AsyncIterator, Iterable, NamedTuple, Sequence
from uuid import UUID

import numpy as np
import sqlalchemy as sql
from pgvector.sqlalchemy import HalfVector  # type: ignore[import-untyped]

from pixeltable import catalog, exprs
from pixeltable.env import Env
from pixeltable.utils.progress_reporter import ProgressReporter

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

if TYPE_CHECKING:
    import pixeltable.plan
    from pixeltable.plan import SampleClause

_logger = logging.getLogger('pixeltable')


class OrderByItem(NamedTuple):
    expr: exprs.Expr
    asc: bool | None


OrderByClause = list[OrderByItem]


def combine_order_by_clauses(clauses: Iterable[OrderByClause]) -> OrderByClause | None:
    """Returns a clause that's compatible with 'clauses', or None if that doesn't exist.
    Two clauses are compatible if for each of their respective items c1[i] and c2[i]
    a) the exprs are identical and
    b) the asc values are identical or at least one is None (None serves as a wildcard)
    """
    result: OrderByClause = []
    for clause in clauses:
        combined: OrderByClause = []
        for item1, item2 in zip(result, clause):
            if item1.expr.id != item2.expr.id:
                return None
            if item1.asc is not None and item2.asc is not None and item1.asc != item2.asc:
                return None
            asc = item1.asc if item1.asc is not None else item2.asc
            combined.append(OrderByItem(item1.expr, asc))

        # add remaining ordering of the longer list
        prefix_len = min(len(result), len(clause))
        if len(result) > prefix_len:
            combined.extend(result[prefix_len:])
        elif len(clause) > prefix_len:
            combined.extend(clause[prefix_len:])
        result = combined
    return result


def print_order_by_clause(clause: OrderByClause) -> str:
    return ', '.join(
        f'({item.expr}{", asc=True" if item.asc is True else ""}{", asc=False" if item.asc is False else ""})'
        for item in clause
    )


class SqlNode(ExecNode):
    """
    Base class for execution nodes that materialize data via SQL queries.

    SqlNode is the primary interface between Pixeltable's execution engine and the
    underlying PostgreSQL database. It generates SQL SELECT statements and converts
    database results into DataRow objects.

    Architecture:
        SqlNode handles the "SQL-materializable" portion of query execution:
        - Column values that can be fetched directly from the database
        - Expressions that PostgreSQL can compute (arithmetic, string ops, etc.)
        - WHERE clause predicates that can be pushed to SQL

        Expressions that can't be computed in SQL (UDFs, Python functions) are
        handled by downstream ExprEvalNode.

    Subclasses:
        - SqlScanNode: Single table SELECT with WHERE, ORDER BY, LIMIT
        - SqlJoinNode: Multi-table SELECT with JOINs
        - SqlAggregationNode: SELECT with GROUP BY aggregation
        - SqlSampleNode: Random sampling queries

    Data Flow:
        1. _create_stmt() builds the SQL SELECT statement
        2. __aiter__() executes the query and yields DataRowBatch objects
        3. _populate_row() converts SQL result rows into DataRow objects

    Key Features:
        - Primary key retrieval (set_pk=True) for row identification
        - Python post-filtering (py_filter) for non-SQL predicates
        - CTE generation (to_cte()) for use in subqueries
        - Progress reporting during long-running queries
    """

    # === Configuration ===
    # Target table (None for joins)
    tbl: catalog.TableVersionPath | None
    # Expressions to fetch via SQL SELECT
    select_list: exprs.ExprSet
    # Columns to populate in DataRow.cell_vals/cell_md
    columns: list[catalog.Column]  # for which columns to populate DataRow.cell_vals/cell_md
    # Column properties that need cellmd loaded for evaluation
    cell_md_refs: list[exprs.ColumnPropertyRef]  # of ColumnRefs which also need DataRow.slot_cellmd for evaluation
    # Whether to retrieve and set primary key on each DataRow
    set_pk: bool
    # Number of primary key columns (rowid columns + version)
    num_pk_cols: int
    # Python-only filter predicate (evaluated after SQL results)
    py_filter: exprs.Expr | None  # a predicate that can only be run in Python
    # Evaluation context for py_filter
    py_filter_eval_ctx: exprs.RowBuilder.EvalCtx | None
    # Cached CTE representation for subquery use
    cte: sql.CTE | None
    # Cache determining which expressions are SQL-materializable
    sql_elements: exprs.SqlElementCache
    # Progress reporting for long queries
    progress_reporter: ProgressReporter | None

    # === Execution State ===
    # Final SQL select list after expansion
    sql_select_list_exprs: exprs.ExprSet
    # Maps cellmd expressions to their index in SQL select list
    cellmd_item_idxs: exprs.ExprDict[int]  # cellmd expr -> idx in sql select list
    # Maps columns to their index in SQL select list
    column_item_idxs: dict[catalog.Column, int]  # column -> idx in sql select list
    column_cellmd_item_idxs: dict[catalog.Column, int]  # column -> idx in sql select list
    result_cursor: sql.engine.CursorResult | None

    # where_clause/-_element: allow subclass to set one or the other (but not both)
    where_clause: exprs.Expr | None
    where_clause_element: sql.ColumnElement | None

    order_by_clause: OrderByClause
    limit: int | None
    offset: int | None

    def __init__(
        self,
        tbl: catalog.TableVersionPath | None,
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        columns: list[catalog.Column],
        sql_elements: exprs.SqlElementCache,
        cell_md_col_refs: list[exprs.ColumnRef] | None = None,
        set_pk: bool = False,
    ):
        # create Select stmt
        self.sql_elements = sql_elements
        self.tbl = tbl
        self.progress_reporter = None
        self.columns = columns
        if cell_md_col_refs is not None:
            assert all(ref.col.stores_cellmd for ref in cell_md_col_refs)
            self.cell_md_refs = [
                exprs.ColumnPropertyRef(ref, exprs.ColumnPropertyRef.Property.CELLMD) for ref in cell_md_col_refs
            ]
        else:
            self.cell_md_refs = []
        self.select_list = exprs.ExprSet(select_list)
        # unstored iter columns: we also need to retrieve whatever is needed to materialize the
        # iter args and stored outputs
        for iter_arg in row_builder.unstored_iter_args.values():
            sql_subexprs = iter_arg.subexprs(filter=self.sql_elements.contains, traverse_matches=False)
            self.select_list.update(sql_subexprs)
        # We query for unstored outputs only if we're not loading a view; when we're loading a view, we are populating
        # those columns, so we need to keep them out of the select list. This isn't a problem, because view loads never
        # need to call set_pos().
        # TODO: This is necessary because create_view_load_plan passes stored output columns to `RowBuilder` via the
        #     `columns` parameter (even though they don't appear in `output_exprs`). This causes them to be recorded as
        #     expressions in `RowBuilder`, which creates a conflict if we add them here. If `RowBuilder` is restructured
        #     to keep them out of `unique_exprs`, then this conditional can be removed.
        if not row_builder.for_view_load:
            for outputs in row_builder.unstored_iter_outputs.values():
                self.select_list.update(outputs)
        super().__init__(row_builder, self.select_list, [], None)  # we materialize self.select_list

        if tbl is not None:
            # minimize the number of tables that need to be joined to the target table
            self.retarget_rowid_refs(tbl, self.select_list)

        assert self.sql_elements.contains_all(self.select_list)
        self.set_pk = set_pk
        self.num_pk_cols = 0
        if set_pk:
            # we also need to retrieve the pk columns
            assert tbl is not None
            self.num_pk_cols = len(tbl.tbl_version.get().store_tbl.pk_columns())
            assert self.num_pk_cols > 1

        # additional state
        self.cellmd_item_idxs = exprs.ExprDict()
        self.column_item_idxs = {}
        self.column_cellmd_item_idxs = {}
        self.result_cursor = None
        # the filter is provided by the subclass
        self.py_filter = None
        self.py_filter_eval_ctx = None
        self.cte = None
        self.limit = None
        self.offset = None
        self.where_clause = None
        self.where_clause_element = None
        self.order_by_clause = []

        if self.tbl is not None:
            tv = self.tbl.tbl_version._tbl_version
            if tv is not None:
                assert tv.is_validated

    def _open(self) -> None:
        desc = 'Rows read'
        if self.tbl is not None:
            desc += f' (table {self.tbl.tbl_name()!r})'
        self.progress_reporter = self.ctx.add_progress_reporter(desc, 'rows')

    def _pk_col_items(self) -> list[sql.Column]:
        if self.set_pk:
            # we need to retrieve the pk columns
            assert self.tbl is not None
            assert self.tbl.tbl_version.get().is_validated
            return self.tbl.tbl_version.get().store_tbl.pk_columns()
        return []

    def _init_exec_state(self) -> None:
        assert self.sql_elements.contains_all(self.select_list)
        self.sql_select_list_exprs = exprs.ExprSet(self.select_list)
        self.cellmd_item_idxs = exprs.ExprDict((ref, self.sql_select_list_exprs.add(ref)) for ref in self.cell_md_refs)
        column_refs = [exprs.ColumnRef(col) for col in self.columns]
        self.column_item_idxs = {col_ref.col: self.sql_select_list_exprs.add(col_ref) for col_ref in column_refs}
        column_cellmd_refs = [
            exprs.ColumnPropertyRef(col_ref, exprs.ColumnPropertyRef.Property.CELLMD)
            for col_ref in column_refs
            if col_ref.col.stores_cellmd
        ]
        self.column_cellmd_item_idxs = {
            cellmd_ref.col_ref.col: self.sql_select_list_exprs.add(cellmd_ref) for cellmd_ref in column_cellmd_refs
        }

    def _create_stmt(self) -> sql.Select:
        """Create Select from local state"""
        self._init_exec_state()
        sql_select_list = [self.sql_elements.get(e) for e in self.sql_select_list_exprs] + self._pk_col_items()
        stmt = sql.select(*sql_select_list)

        where_clause_element = (
            self.sql_elements.get(self.where_clause) if self.where_clause is not None else self.where_clause_element
        )
        if where_clause_element is not None:
            stmt = stmt.where(where_clause_element)

        order_by_clause: list[sql.ColumnElement] = []
        for e, asc in self.order_by_clause:
            if isinstance(e, exprs.SimilarityExpr):
                order_by_clause.append(e.as_order_by_clause(asc))
            else:
                order_by_clause.append(self.sql_elements.get(e).desc() if asc is False else self.sql_elements.get(e))
        stmt = stmt.order_by(*order_by_clause)

        if self.py_filter is None:
            # if we don't have a Python filter, we can apply limit/offset to stmt
            if self.limit is not None:
                stmt = stmt.limit(self.limit)
            if self.offset is not None:
                stmt = stmt.offset(self.offset)

        return stmt

    def _ordering_tbl_ids(self) -> set[UUID]:
        return exprs.Expr.all_tbl_ids(e for e, _ in self.order_by_clause)

    def to_cte(self, keep_pk: bool = False) -> tuple[sql.CTE, exprs.ExprDict[sql.ColumnElement]] | None:
        """
        Creates a CTE that materializes the output of this node plus a mapping from select list expr to output column.
        keep_pk: if True, the PK columns are included in the CTE Select statement

        Returns:
            (CTE, dict from Expr to output column)
        """
        if self.py_filter is not None:
            # the filter needs to run in Python
            return None
        if self.cte is None:
            if not keep_pk:
                self.set_pk = False  # we don't need the PK if we use this SqlNode as a CTE
            self.cte = self._create_stmt().cte()
        return self.cte, exprs.ExprDict(zip(list(self.select_list) + self.cell_md_refs, self.cte.c))  # skip pk cols

    @classmethod
    def retarget_rowid_refs(cls, target: catalog.TableVersionPath, expr_seq: Iterable[exprs.Expr]) -> None:
        """Change rowid refs to point to target"""
        for e in expr_seq:
            if isinstance(e, exprs.RowidRef):
                e.set_tbl(target)

    @classmethod
    def create_from_clause(
        cls,
        tbl: catalog.TableVersionPath,
        stmt: sql.Select,
        refd_tbl_ids: set[UUID] | None = None,
        exact_version_only: set[UUID] | None = None,
    ) -> sql.Select:
        """Add From clause to stmt for tables/views referenced by materialized_exprs
        Args:
            tbl: root table of join chain
            stmt: stmt to add From clause to
            materialized_exprs: list of exprs that reference tables in the join chain; if empty, include only the root
            exact_version_only: set of table ids for which we only want to see rows created at the current version
        Returns:
            augmented stmt
        """
        # we need to include at least the root
        if refd_tbl_ids is None:
            refd_tbl_ids = set()
        if exact_version_only is None:
            exact_version_only = set()
        candidates = tbl.get_tbl_versions()
        assert len(candidates) > 0
        joined_tbls: list[catalog.TableVersionHandle] = [candidates[0]]
        for t in candidates[1:]:
            if t.id in refd_tbl_ids:
                joined_tbls.append(t)

        first = True
        prev_tv: catalog.TableVersion | None = None
        for t in joined_tbls[::-1]:
            tv = t.get()
            # _logger.debug(f'create_from_clause: tbl_id={tv.id} {id(tv.store_tbl.sa_tbl)}')
            if first:
                stmt = stmt.select_from(tv.store_tbl.sa_tbl)
                first = False
            else:
                # join tv to prev_tv on prev_tv's rowid cols
                prev_tbl_rowid_cols = prev_tv.store_tbl.rowid_columns()
                tbl_rowid_cols = tv.store_tbl.rowid_columns()
                rowid_clauses = [
                    c1 == c2 for c1, c2 in zip(prev_tbl_rowid_cols, tbl_rowid_cols[: len(prev_tbl_rowid_cols)])
                ]
                stmt = stmt.join(tv.store_tbl.sa_tbl, sql.and_(*rowid_clauses))

            if t.id in exact_version_only:
                stmt = stmt.where(tv.store_tbl.v_min_col == tv.version)
            else:
                stmt = stmt.where(tv.store_tbl.sa_tbl.c.v_min <= tv.version)
                stmt = stmt.where(tv.store_tbl.sa_tbl.c.v_max > tv.version)
            prev_tv = tv

        return stmt

    def set_where(self, where_clause: exprs.Expr) -> None:
        assert self.where_clause_element is None
        self.where_clause = where_clause

    def set_py_filter(self, py_filter: exprs.Expr) -> None:
        assert self.py_filter is None
        self.py_filter = py_filter
        self.py_filter_eval_ctx = self.row_builder.create_eval_ctx([py_filter], exclude=self.select_list)

    def set_order_by(self, ordering: OrderByClause) -> None:
        """Add Order By clause"""
        if self.tbl is not None:
            # change rowid refs against a base table to rowid refs against the target table, so that we minimize
            # the number of tables that need to be joined to the target table
            self.retarget_rowid_refs(self.tbl, [e for e, _ in ordering])
        combined = combine_order_by_clauses([self.order_by_clause, ordering])
        assert combined is not None
        self.order_by_clause = combined

    def set_limit(self, limit: int) -> None:
        self.limit = limit

    def set_offset(self, offset: int) -> None:
        self.offset = offset

    def _log_explain(self, stmt: sql.Select) -> None:
        conn = Env.get().conn
        try:
            # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
            stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
            explain_result = conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
            explain_str = '\n'.join([str(row) for row in explain_result])
            _logger.debug(f'SqlScanNode explain:\n{explain_str}')
        except Exception as e:
            _logger.warning(f'EXPLAIN failed with error: {e}')

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        # run the query; do this here rather than in _open(), exceptions are only expected during iteration
        with warnings.catch_warnings(record=True) as w:
            stmt = self._create_stmt()
            try:
                # log stmt, if possible
                stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
                _logger.debug(f'SqlLookupNode stmt:\n{stmt_str}')
            except Exception:
                # log something if we can't log the compiled stmt
                _logger.debug(f'SqlLookupNode proto-stmt:\n{stmt}')
            self._log_explain(stmt)

            conn = Env.get().conn
            result_cursor = conn.execute(stmt)
            for _ in w:
                pass

        output_batch = DataRowBatch(self.row_builder)
        output_row: exprs.DataRow | None = None
        num_rows_read = 0
        is_using_cockroachdb = Env.get().is_using_cockroachdb
        tzinfo = Env.get().default_time_zone

        for sql_row in result_cursor:
            output_row = output_batch.add_row(output_row)

            # populate output_row

            if self.num_pk_cols > 0:
                output_row.set_pk(tuple(sql_row[-self.num_pk_cols :]))

            # column copies
            for col, item_idx in self.column_item_idxs.items():
                output_row.cell_vals[col.id] = sql_row[item_idx]
            for col, item_idx in self.column_cellmd_item_idxs.items():
                cell_md_dict = sql_row[item_idx]
                output_row.cell_md[col.id] = exprs.CellMd(**cell_md_dict) if cell_md_dict is not None else None

            # populate DataRow.slot_cellmd, where requested
            for cellmd_ref, item_idx in self.cellmd_item_idxs.items():
                cell_md_dict = sql_row[item_idx]
                output_row.slot_md[cellmd_ref.col_ref.slot_idx] = (
                    exprs.CellMd.from_dict(cell_md_dict) if cell_md_dict is not None else None
                )

            # copy the output of the SQL query into the output row
            for i, e in enumerate(self.select_list):
                slot_idx = e.slot_idx
                if isinstance(sql_row[i], Decimal):
                    # certain numerical operations can produce Decimals (eg, SUM(<int column>)); we need to convert them
                    if e.col_type.is_int_type():
                        output_row[slot_idx] = int(sql_row[i])
                    elif e.col_type.is_float_type():
                        output_row[slot_idx] = float(sql_row[i])
                    else:
                        raise RuntimeError(f'Unexpected Decimal value for {e}')
                elif is_using_cockroachdb and isinstance(sql_row[i], datetime.datetime):
                    # Ensure that the datetime is timezone-aware and in the session time zone
                    # cockroachDB returns timestamps in the session time zone, with numeric offset,
                    # convert to the session time zone with the requested tzinfo for DST handling
                    if e.col_type.is_timestamp_type():
                        if isinstance(sql_row[i].tzinfo, datetime.timezone):
                            output_row[slot_idx] = sql_row[i].astimezone(tz=tzinfo)
                        else:
                            output_row[slot_idx] = sql_row[i]
                    else:
                        raise RuntimeError(f'Unexpected datetime value for {e}')
                elif isinstance(sql_row[i], HalfVector):
                    # All array data needs to be materialized as ndarrays
                    output_row[slot_idx] = sql_row[i].to_numpy().astype(np.float32)
                else:
                    output_row[slot_idx] = sql_row[i]

            if self.py_filter is not None:
                # evaluate filter
                self.row_builder.eval(output_row, self.py_filter_eval_ctx, profile=self.ctx.profile)
                if not output_row[self.py_filter.slot_idx]:
                    # didn't pass filter; re-use this row for the next sql row
                    output_row = output_batch.pop_row()
                    output_row.clear()
                    continue

            # Row passed filter (or no filter)
            num_rows_read += 1

            # if we're using a Python filter, we need to apply offset/limit logic here. (with a SQL filter
            # that logic has already been baked into the query)
            if self.py_filter is not None:
                # Check if we should skip this row due to offset
                if self.offset is not None and num_rows_read <= self.offset:
                    # Skip this row - remove it from batch
                    output_row = output_batch.pop_row()
                    output_row.clear()
                    continue

                # Check if we've reached the limit (after offset)
                if self.limit is not None:
                    num_rows_returned = num_rows_read - (self.offset or 0)
                    assert num_rows_returned <= self.limit
                    if num_rows_returned == self.limit:
                        break

            # Include this row in output
            output_row = None

            if self.ctx.batch_size > 0 and len(output_batch) == self.ctx.batch_size:
                _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
                if self.progress_reporter is not None:
                    self.progress_reporter.update(len(output_batch))
                yield output_batch
                output_batch = DataRowBatch(self.row_builder)

        if len(output_batch) > 0:
            _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
            if self.progress_reporter is not None:
                self.progress_reporter.update(len(output_batch))
            yield output_batch

    def _close(self) -> None:
        if self.result_cursor is not None:
            self.result_cursor.close()


class SqlScanNode(SqlNode):
    """
    Fetches rows from a single table via SQL SELECT.

    SqlScanNode is the most common source node in execution plans. It generates
    SQL SELECT statements that can include:
    - Column selections and expressions
    - WHERE clause filtering (pushed from Analyzer.sql_where_clause)
    - ORDER BY sorting
    - LIMIT/OFFSET pagination

    The FROM clause is constructed to include all necessary table joins for
    the selected expressions (e.g., joining to base tables for view columns).

    Example SQL generated:
        SELECT col_a, col_b, col_a + col_b AS expr_0
        FROM my_table t
        WHERE t.col_a > 5
        ORDER BY t.col_a
        LIMIT 100

    Args:
        tbl: Table to scan
        select_list: Expressions to materialize
        set_pk: Whether to retrieve primary key columns
        exact_version_only: Tables for which to only see rows at current version
    """

    # Tables that should only return rows from the exact current version
    exact_version_only: list[catalog.TableVersionHandle]

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        columns: list[catalog.Column],
        cell_md_col_refs: list[exprs.ColumnRef] | None = None,
        set_pk: bool = False,
        exact_version_only: list[catalog.TableVersionHandle] | None = None,
    ):
        sql_elements = exprs.SqlElementCache()
        super().__init__(
            tbl,
            row_builder,
            select_list,
            columns=columns,
            sql_elements=sql_elements,
            set_pk=set_pk,
            cell_md_col_refs=cell_md_col_refs,
        )
        # create Select stmt
        if exact_version_only is None:
            exact_version_only = []

        self.exact_version_only = exact_version_only

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        where_clause_tbl_ids = self.where_clause.tbl_ids() if self.where_clause is not None else set()
        refd_tbl_ids = exprs.Expr.all_tbl_ids(self.select_list) | where_clause_tbl_ids | self._ordering_tbl_ids()
        stmt = self.create_from_clause(
            self.tbl, stmt, refd_tbl_ids, exact_version_only={t.id for t in self.exact_version_only}
        )
        return stmt


class SqlLookupNode(SqlNode):
    """
    Materializes data from the store via a Select stmt with a WHERE clause that matches a list of key values

    Args:
        select_list: output of the query
        sa_key_cols: list of key columns in the store table
        key_vals: list of key values to look up
    """

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        columns: list[catalog.Column],
        sa_key_cols: list[sql.Column],
        key_vals: list[tuple],
        cell_md_col_refs: list[exprs.ColumnRef] | None = None,
    ):
        sql_elements = exprs.SqlElementCache()
        super().__init__(
            tbl,
            row_builder,
            select_list,
            columns=columns,
            sql_elements=sql_elements,
            set_pk=True,
            cell_md_col_refs=cell_md_col_refs,
        )
        # Where clause: (key-col-1, key-col-2, ...) IN ((val-1, val-2, ...), ...)
        self.where_clause_element = sql.tuple_(*sa_key_cols).in_(key_vals)

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        refd_tbl_ids = exprs.Expr.all_tbl_ids(self.select_list) | self._ordering_tbl_ids()
        stmt = self.create_from_clause(self.tbl, stmt, refd_tbl_ids)
        return stmt


class SqlAggregationNode(SqlNode):
    """
    Performs GROUP BY aggregation in SQL.

    Used when all aggregation expressions (sum, count, avg, etc.) can be computed
    by PostgreSQL. Takes a SqlNode as input (via CTE) and adds GROUP BY clause.

    When SQL aggregation isn't possible (e.g., custom Python aggregates),
    the Planner uses AggregationNode instead.

    Example SQL generated:
        WITH input_cte AS (SELECT ... FROM ...)
        SELECT group_col, SUM(value_col), COUNT(*)
        FROM input_cte
        GROUP BY group_col

    Args:
        input: Source SqlNode (converted to CTE)
        select_list: Expressions including aggregate function calls
        group_by_items: Expressions to group by (None for single-group aggregation)
    """

    # GROUP BY expressions (None = single group for entire result)
    group_by_items: list[exprs.Expr] | None
    # Input query as CTE
    input_cte: sql.CTE | None

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        input: SqlNode,
        select_list: Iterable[exprs.Expr],
        group_by_items: list[exprs.Expr] | None = None,
        limit: int | None = None,
        exact_version_only: list[catalog.TableVersion] | None = None,
    ):
        assert len(input.cell_md_refs) == 0  # there's no aggregation over json or arrays in SQL
        self.input_cte, input_col_map = input.to_cte()
        sql_elements = exprs.SqlElementCache(input_col_map)
        super().__init__(None, row_builder, select_list, columns=[], sql_elements=sql_elements)
        self.group_by_items = group_by_items

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt().select_from(self.input_cte)
        if self.group_by_items is not None:
            sql_group_by_items = [self.sql_elements.get(e) for e in self.group_by_items]
            assert all(e is not None for e in sql_group_by_items)
            stmt = stmt.group_by(*sql_group_by_items)
        return stmt


class SqlJoinNode(SqlNode):
    """
    Joins multiple tables via SQL JOIN clauses.

    Takes multiple SqlScanNode inputs (one per table) and combines them using
    SQL JOIN operations. Each input is converted to a CTE, then joined according
    to the join_clauses specification.

    Supports:
    - INNER JOIN
    - LEFT OUTER JOIN
    - FULL OUTER JOIN
    - CROSS JOIN

    Example SQL generated:
        WITH cte0 AS (SELECT ... FROM table1),
             cte1 AS (SELECT ... FROM table2)
        SELECT cte0.col_a, cte1.col_b
        FROM cte0
        JOIN cte1 ON cte0.id = cte1.foreign_id

    Args:
        inputs: List of SqlScanNode objects, one per table
        join_clauses: Join specifications (type and predicate) for each join
        select_list: Expressions to return from the joined result
    """

    # Input queries converted to CTEs
    input_ctes: list[sql.CTE]
    # Join specifications (join_clauses[i] joins inputs[i] with inputs[i+1])
    join_clauses: list['pixeltable.plan.JoinClause']

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        inputs: Sequence[SqlNode],
        join_clauses: list['pixeltable.plan.JoinClause'],
        select_list: Iterable[exprs.Expr],
    ):
        assert len(inputs) > 1
        assert len(inputs) == len(join_clauses) + 1
        self.input_ctes = []
        self.join_clauses = join_clauses
        sql_elements = exprs.SqlElementCache()
        for input_node in inputs:
            input_cte, input_col_map = input_node.to_cte()
            self.input_ctes.append(input_cte)
            sql_elements.extend(input_col_map)
        cell_md_col_refs = [cell_md_ref.col_ref for input in inputs for cell_md_ref in input.cell_md_refs]
        super().__init__(
            None, row_builder, select_list, columns=[], sql_elements=sql_elements, cell_md_col_refs=cell_md_col_refs
        )

    def _create_stmt(self) -> sql.Select:
        from pixeltable import plan

        stmt = super()._create_stmt()
        stmt = stmt.select_from(self.input_ctes[0])
        for i in range(len(self.join_clauses)):
            join_clause = self.join_clauses[i]
            on_clause = (
                self.sql_elements.get(join_clause.join_predicate)
                if join_clause.join_type != plan.JoinType.CROSS
                else sql.sql.expression.literal(True)
            )
            is_outer = join_clause.join_type in (plan.JoinType.LEFT, plan.JoinType.FULL_OUTER)
            stmt = stmt.join(
                self.input_ctes[i + 1],
                onclause=on_clause,
                isouter=is_outer,
                full=join_clause == plan.JoinType.FULL_OUTER,
            )
        return stmt


class SqlSampleNode(SqlNode):
    """
    Returns rows sampled from the input node.

    Args:
        input: SqlNode to sample from
        select_list: can contain calls to AggregateFunctions
        sample_clause: specifies the sampling method
        stratify_exprs: Analyzer processed list of expressions to stratify by.
    """

    input_cte: sql.CTE | None
    pk_count: int
    stratify_exprs: list[exprs.Expr] | None
    sample_clause: 'SampleClause'

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        input: SqlNode,
        select_list: Iterable[exprs.Expr],
        sample_clause: 'SampleClause',
        stratify_exprs: list[exprs.Expr],
    ):
        assert isinstance(input, SqlNode)
        self.input_cte, input_col_map = input.to_cte(keep_pk=True)
        self.pk_count = input.num_pk_cols
        assert self.pk_count > 1
        sql_elements = exprs.SqlElementCache(input_col_map)
        assert sql_elements.contains_all(stratify_exprs)
        cell_md_col_refs = [cell_md_ref.col_ref for cell_md_ref in input.cell_md_refs]
        super().__init__(
            input.tbl,
            row_builder,
            select_list,
            columns=[],
            sql_elements=sql_elements,
            cell_md_col_refs=cell_md_col_refs,
            set_pk=True,
        )
        self.stratify_exprs = stratify_exprs
        self.sample_clause = sample_clause

    @classmethod
    def key_sql_expr(cls, seed: sql.ColumnElement, sql_cols: Iterable[sql.ColumnElement]) -> sql.ColumnElement:
        """Construct expression which is the ordering key for rows to be sampled
        General SQL form is:
        - MD5(<seed::text> [ + '___' + <rowid_col_val>::text]+
        """
        sql_expr: sql.ColumnElement = seed.cast(sql.String)
        for e in sql_cols:
            # Quotes are required below to guarantee that the string is properly presented in SQL
            sql_expr = sql_expr + sql.literal_column("'___'", sql.Text) + e.cast(sql.String)
        sql_expr = sql.func.md5(sql_expr)
        return sql_expr

    def _create_key_sql(self, cte: sql.CTE) -> sql.ColumnElement:
        """Create an expression for randomly ordering rows with a given seed"""
        rowid_cols = [*cte.c[-self.pk_count : -1]]  # exclude the version column
        assert len(rowid_cols) > 0
        # If seed is not set in the sample clause, use the random seed given by the execution context
        seed = self.sample_clause.seed if self.sample_clause.seed is not None else self.ctx.random_seed
        return self.key_sql_expr(sql.literal_column(str(seed)), rowid_cols)

    def _create_stmt(self) -> sql.Select:
        from pixeltable.plan import SampleClause

        self._init_exec_state()

        if self.sample_clause.fraction is not None:
            if len(self.stratify_exprs) == 0:
                # If non-stratified sampling, construct a where clause, order_by, and limit clauses
                s_key = self._create_key_sql(self.input_cte)

                # Construct a suitable where clause
                fraction_md5 = SampleClause.fraction_to_md5_hex(self.sample_clause.fraction)
                order_by = self._create_key_sql(self.input_cte)
                return sql.select(*self.input_cte.c).where(s_key < fraction_md5).order_by(order_by)

            return self._create_stmt_stratified_fraction(self.sample_clause.fraction)
        else:
            if len(self.stratify_exprs) == 0:
                # No stratification, just return n samples from the input CTE
                order_by = self._create_key_sql(self.input_cte)
                return sql.select(*self.input_cte.c).order_by(order_by).limit(self.sample_clause.n)

            return self._create_stmt_stratified_n(self.sample_clause.n, self.sample_clause.n_per_stratum)

    def _create_stmt_stratified_n(self, n: int | None, n_per_stratum: int | None) -> sql.Select:
        """Create a Select stmt that returns n samples across all strata or n_per_stratum samples per stratum"""

        sql_strata_exprs = [self.sql_elements.get(e) for e in self.stratify_exprs]
        order_by = self._create_key_sql(self.input_cte)

        # Create a list of all columns plus the rank
        # Get all columns from the input CTE dynamically
        select_columns = [*self.input_cte.c]
        select_columns.append(
            sql.func.row_number().over(partition_by=sql_strata_exprs, order_by=order_by).label('rank')
        )
        row_rank_cte = sql.select(*select_columns).select_from(self.input_cte).cte('row_rank_cte')

        final_columns = [*row_rank_cte.c[:-1]]  # exclude the rank column
        if n_per_stratum is not None:
            return sql.select(*final_columns).filter(row_rank_cte.c.rank <= n_per_stratum)
        else:
            secondary_order = self._create_key_sql(row_rank_cte)
            return sql.select(*final_columns).order_by(row_rank_cte.c.rank, secondary_order).limit(n)

    def _create_stmt_stratified_fraction(self, fraction_samples: float) -> sql.Select:
        """Create a Select stmt that returns a fraction of the rows per strata"""

        # Build the strata count CTE
        # Produces a table of the form:
        #   (*stratify_exprs, s_s_size)
        # where s_s_size is the number of samples to take from each stratum
        sql_strata_exprs = [self.sql_elements.get(e) for e in self.stratify_exprs]
        per_strata_count_cte = (
            sql.select(
                *sql_strata_exprs,
                sql.func.ceil(fraction_samples * sql.func.count(1).cast(sql.Integer)).label('s_s_size'),
            )
            .select_from(self.input_cte)
            .group_by(*sql_strata_exprs)
            .cte('per_strata_count_cte')
        )

        # Build a CTE that ranks the rows within each stratum
        # Include all columns from the input CTE dynamically
        order_by = self._create_key_sql(self.input_cte)
        select_columns = [*self.input_cte.c]
        select_columns.append(
            sql.func.row_number().over(partition_by=sql_strata_exprs, order_by=order_by).label('rank')
        )
        row_rank_cte = sql.select(*select_columns).select_from(self.input_cte).cte('row_rank_cte')

        # Build the join criterion dynamically to accommodate any number of stratify_by expressions
        join_c = sql.true()
        for col in per_strata_count_cte.c[:-1]:
            join_c &= row_rank_cte.c[col.name].isnot_distinct_from(col)

        # Join with per_strata_count_cte to limit returns to the requested fraction of rows
        final_columns = [*row_rank_cte.c[:-1]]  # exclude the rank column
        stmt = (
            sql.select(*final_columns)
            .select_from(row_rank_cte)
            .join(per_strata_count_cte, join_c)
            .where(row_rank_cte.c.rank <= per_strata_count_cte.c.s_s_size)
        )

        return stmt

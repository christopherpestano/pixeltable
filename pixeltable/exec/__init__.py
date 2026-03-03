"""
Query execution engine for Pixeltable.

This package implements the execution pipeline that transforms query plans into results.
The engine uses a tree of ExecNode objects that process data as DataRowBatch objects flowing
from source nodes (SqlScanNode, InMemoryDataNode) through transformation nodes (ExprEvalNode,
AggregationNode) to the caller.

Key components:
    - ExecNode: Abstract base class for all execution nodes
    - SqlNode and subclasses: Fetch data from PostgreSQL via SQL queries
    - ExprEvalNode: Evaluate computed columns, UDFs, and expressions in Python
    - DataRowBatch: Container for rows flowing between execution nodes
    - ExecContext: Runtime context with progress reporting and configuration

The expr_eval subpackage contains the expression evaluation machinery, including
evaluators for different expression types, schedulers for rate-limited API calls,
and a row buffer for maintaining output order.
"""

# ruff: noqa: F401

from .aggregation_node import AggregationNode
from .cache_prefetch_node import CachePrefetchNode
from .cell_materialization_node import CellMaterializationNode
from .cell_reconstruction_node import CellReconstructionNode
from .component_iteration_node import ComponentIterationNode
from .data_row_batch import DataRowBatch
from .exec_context import ExecContext
from .exec_node import ExecNode
from .expr_eval import ExprEvalNode
from .in_memory_data_node import InMemoryDataNode
from .object_store_save_node import ObjectStoreSaveNode
from .row_update_node import RowUpdateNode
from .sql_node import SqlAggregationNode, SqlJoinNode, SqlLookupNode, SqlNode, SqlSampleNode, SqlScanNode

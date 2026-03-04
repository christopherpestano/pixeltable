"""
pixeltable.catalog - Core schema management and data model for Pixeltable.

This package implements the catalog layer, which manages Tables, Views, Columns,
and their metadata. It is the heart of Pixeltable's schema management, providing:

- **Schema objects**: Tables (InsertableTable), Views, Snapshots, Directories
- **Versioning**: Every schema/data change creates a new TableVersion
- **Column management**: Computed columns, indices, media validation
- **Transaction coordination**: Optimistic concurrency with retry logic
- **Metadata persistence**: All schema metadata stored in PostgreSQL via SQLAlchemy

The module is organized around a layered indirection pattern:
  Table -> TableVersionPath -> TableVersionHandle -> TableVersion
This allows safe cross-transaction references and lazy metadata loading.

Key entry points:
- Catalog: The central registry that manages all schema objects
- Table/InsertableTable/View: User-facing handles for data operations
- Column: Represents a column with its type, computed expression, and storage info
"""

# ruff: noqa: F401

from .catalog import Catalog, retry_loop
from .column import Column
from .dir import Dir
from .globals import IfExistsParam, IfNotExistsParam, MediaValidation, QColumnId, is_valid_identifier
from .insertable_table import InsertableTable
from .path import Path
from .schema_object import SchemaObject
from .table import Table
from .table_metadata import ColumnMetadata, IndexMetadata, TableMetadata, VersionMetadata
from .table_version import TableVersion
from .table_version_handle import ColumnHandle, TableVersionHandle
from .table_version_path import TableVersionPath
from .update_status import RowCountStats, UpdateStatus
from .view import View

# pixeltable/catalog/ - Core Schema Management

The catalog module is the heart of Pixeltable's data model. It manages the lifecycle
of tables, views, columns, indices, and directories, providing schema management,
versioning, transactional coordination, and metadata persistence.

## Architecture Overview

```
                    User API (pxt.create_table, t.select, t.insert, ...)
                                        |
                    +-------------------v-------------------+
                    |              Catalog                   |
                    |  (singleton registry & coordinator)    |
                    |  - Transaction management (begin_xact) |
                    |  - Retry logic (retry_loop)            |
                    |  - Metadata caching & invalidation     |
                    |  - Directory tree management           |
                    |  - Column dependency tracking          |
                    +-------------------+-------------------+
                                        |
              +-------------------------+-------------------------+
              |                         |                         |
    +---------v---------+   +-----------v-----------+   +---------v---------+
    |   InsertableTable |   |         View          |   |        Dir        |
    |   (base tables)   |   |  (views, snapshots,   |   |   (directories)   |
    |   insert/delete   |   |  component views)     |   |                   |
    +---------+---------+   +-----------+-----------+   +-------------------+
              |                         |
              +------------+------------+
                           |
                  +--------v--------+
                  |      Table      |
                  | (query iface +  |
                  |  schema ops)    |
                  +--------+--------+
                           |
              +------------v------------+
              |    TableVersionPath     |
              | (linked list of handles |
              |  from view to root)     |
              +------------+------------+
                           |
              +------------v------------+
              |   TableVersionHandle    |
              | (lazy indirection to    |
              |  TableVersion via cache)|
              +------------+------------+
                           |
              +------------v------------+
              |     TableVersion        |
              | (versioned schema +     |
              |  physical store ref)    |
              +--+--+--+--+--+---------+
                 |  |  |  |  |
       +---------+  |  |  |  +----------+
       |            |  |  |             |
  +----v----+  +----v--v--v----+  +-----v-----+
  | Column  |  |   IndexInfo   |  | StoreBase |
  | (schema |  | (B-tree,      |  | (physical |
  |  + type)|  |  embedding)   |  |  PG table)|
  +---------+  +---------------+  +-----------+
```

## Class Hierarchy

```
SchemaObject (abstract base)
    |
    +-- Dir (namespace directory)
    |
    +-- Table (query interface + schema operations)
         |
         +-- InsertableTable (base tables: insert, delete)
         |
         +-- View (views, snapshots, component views)
```

## File-by-File Description

### Core Schema Objects

| File | Key Classes | Purpose |
|------|------------|---------|
| `schema_object.py` | `SchemaObject` | Abstract base for all named catalog objects (UUID, name, parent dir) |
| `table.py` | `Table` | User-facing handle: query interface (select/where/join), schema ops (add_column, add_index), metadata display |
| `insertable_table.py` | `InsertableTable` | Extends Table with insert/delete; handles Pydantic models, DataFrames, CSV, etc. |
| `view.py` | `View` | Virtual tables: filtered views, snapshots, component views (iterators), replicas |
| `dir.py` | `Dir` | Namespace directories forming the catalog tree |
| `column.py` | `Column` | Column metadata: type, computed expression, storage config, SA column mappings |

### Versioning & Indirection

| File | Key Classes | Purpose |
|------|------------|---------|
| `table_version.py` | `TableVersion`, `TableVersionMd`, `TableVersionKey` | Core internal class holding versioned schema state, columns, indices, store table reference |
| `table_version_handle.py` | `TableVersionHandle`, `ColumnHandle` | Lazy indirection that survives transaction boundaries; resolves via Catalog cache |
| `table_version_path.py` | `TableVersionPath` | Linked list from a view through its base chain; essential for query joins and column resolution |

### Infrastructure

| File | Key Classes | Purpose |
|------|------------|---------|
| `catalog.py` | `Catalog`, `retry_loop` | Central registry: transactions, locking, caching, directory ops, table create/drop |
| `tbl_ops.py` | `TableOp` and subclasses | Crash-recoverable operation log (rollforward/rollback protocol) |
| `globals.py` | `QColumnId`, `MediaValidation`, `IfExistsParam` | Shared enums, constants, and identifier validation |
| `path.py` | `Path` | Parsed catalog path ('dir.table' or 'dir/table:3') |
| `table_metadata.py` | `TableMetadata`, `ColumnMetadata`, etc. | TypedDict definitions for the public metadata API |
| `update_status.py` | `UpdateStatus`, `RowCountStats` | Result objects returned by mutation operations |

## Key Data Flows

### Table Creation (`pxt.create_table()`)

```
1. Catalog.create_table()
   - Validates path, acquires directory lock
   - Calls InsertableTable._create() to build TableVersionMd
   - Writes metadata + PendingTableOps in a single transaction
2. Catalog._roll_forward()
   - CreateTableMdOp (no-op on exec; undo deletes metadata)
   - CreateStoreTableOp (CREATE TABLE in PostgreSQL)
3. Table instance returned to user
```

### View Creation (`pxt.create_view()`)

```
1. Catalog.create_view()
   - Validates predicates, columns, iterator config against base
   - Writes metadata + PendingTableOps
2. Catalog._roll_forward()
   - CreateTableMdOp
   - CreateStoreTableOp
   - LoadViewOp (populates view by reading matching base rows)
3. View instance returned to user
```

### Insert (`t.insert()`)

```
1. InsertableTable.insert()
   - Normalizes input (dicts, Pydantic, DataFrame, etc.)
   - Acquires write lock on mutable tree
2. TableVersion.insert()
   - Planner builds ExecNode tree
   - StoreBase.insert_rows() writes to PostgreSQL
   - Propagates to mutable views via _propagate_insert()
3. UpdateStatus returned with row counts
```

### Column Addition (`t.add_computed_column()`)

```
1. Table.add_computed_column()
   - Validates expression and column spec
   - Acquires write lock on mutable tree
2. TableVersion.add_columns()
   - Bumps schema version
   - Adds physical column to store table (ALTER TABLE)
   - Populates column values using Planner.create_add_column_plan()
   - Creates B-tree index if applicable
   - Writes updated metadata
```

## Key Design Patterns

### Layered Indirection (Table -> Handle -> TableVersion)

TableVersion instances are invalidated at each transaction boundary because the
underlying metadata may have been changed by a concurrent process. To safely hold
references across transactions, the catalog uses a chain of indirection:

```
Table._tbl_version_path -> TableVersionPath
    .tbl_version -> TableVersionHandle
        .get() -> TableVersion (resolved from Catalog cache)
```

### Crash Recovery (PendingTableOps)

Multi-step mutations (create table, add column) are split into ordered TableOps:
1. All ops are written to PendingTableOp in the initial transaction
2. Ops are executed one at a time with status tracking
3. On crash, next access detects pending ops and resumes (rollforward or rollback)

### Optimistic Concurrency (retry_loop)

Concurrent processes sharing the same Pixeltable instance use optimistic locking:
- `retry_loop()` retries on PostgreSQL SerializationFailure / LockNotAvailable
- `begin_xact()` acquires X-locks on Table records before mutations
- Lock order: parent directory before child, sorted by name to prevent deadlocks

### Column Dependency Tracking

Computed columns can reference columns from the same table or base tables. The
Catalog tracks these dependencies to:
- Propagate updates correctly through the mutable view tree
- Prevent dropping columns that have dependents
- Recompute dependent columns when source data changes

## Relationship to Other Modules

| Module | Relationship |
|--------|-------------|
| `pixeltable.exec` | Query execution engine; Catalog provides schema for plan construction |
| `pixeltable.exprs` | Expression tree; Column.value_expr is an Expr defining computed columns |
| `pixeltable.func` | UDF infrastructure; computed columns use UDFs as value expressions |
| `pixeltable.index` | Index implementations (BtreeIndex, EmbeddingIndex); managed by TableVersion |
| `pixeltable.metadata.schema` | SQLAlchemy ORM models for persistent metadata storage |
| `pixeltable.store` | Physical store tables (StoreTable, StoreView); created/managed by TableVersion |
| `pixeltable.io` | Import/export; InsertableTable delegates to TableDataConduit for data ingestion |
| `pixeltable.plan` | Query planner; uses TableVersionPath for query construction |

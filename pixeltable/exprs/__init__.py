"""
Expression types for Pixeltable's query language.

This module defines the expression tree nodes that form the core of Pixeltable's declarative
computation model. Every computed column, filter predicate, projection, and similarity search
is represented as a tree of :class:`Expr` objects. These expression trees are:

1. **Constructed** by user-facing Python operators (e.g., ``t.col1 + t.col2``, ``t.col == 'foo'``)
2. **Analyzed** by the query planner to assign evaluation slots and determine dependencies
3. **Evaluated** either by translating to SQL (``sql_expr()``) or executing in Python (``eval()``)
4. **Serialized** to JSON for persistence in the catalog metadata

Key base class: :class:`Expr` -- all expression types inherit from it and implement
``eval()``, ``sql_expr()``, ``_as_dict()``/``_from_dict()`` for serialization.

Expression types fall into several categories:

- **Leaf nodes**: ColumnRef, Literal, RowidRef, Variable, ObjectRef
- **Operators**: ArithmeticExpr, Comparison, CompoundPredicate, StringOp, IsNull, InPredicate
- **Data access**: JsonPath, ArraySlice, ColumnPropertyRef
- **Function application**: FunctionCall, MethodRef
- **Inline constructors**: InlineArray, InlineList, InlineDict
- **Specialized**: SimilarityExpr, TypeCast, JsonMapper/JsonMapperDispatch

Supporting infrastructure:

- **DataRow**: The mutable row buffer holding expression values during evaluation
- **RowBuilder**: DAG coordinator that assigns slots, tracks dependencies, and drives evaluation
- **ExprSet / ExprDict**: Identity-based collections using Expr.id (since __eq__ is overloaded)
- **SqlElementCache**: Memoized SQL translation for the expression tree
"""
# ruff: noqa: F401

from .arithmetic_expr import ArithmeticExpr
from .array_slice import ArraySlice
from .column_property_ref import ColumnPropertyRef
from .column_ref import ColumnRef
from .comparison import Comparison
from .compound_predicate import CompoundPredicate
from .data_row import ArrayMd, BinaryMd, CellMd, DataRow
from .expr import Expr
from .expr_dict import ExprDict
from .expr_set import ExprSet
from .function_call import FunctionCall
from .globals import ArithmeticOperator, ComparisonOperator, LogicalOperator
from .in_predicate import InPredicate
from .inline_expr import InlineArray, InlineDict, InlineList
from .is_null import IsNull
from .json_mapper import JsonMapper, JsonMapperDispatch
from .json_path import JsonPath
from .literal import Literal
from .method_ref import MethodRef
from .object_ref import ObjectRef
from .row_builder import ColumnSlotIdx, ExecProfile, RowBuilder
from .rowid_ref import RowidRef
from .similarity_expr import SimilarityExpr
from .sql_element_cache import SqlElementCache
from .string_op import StringOp
from .type_cast import TypeCast
from .variable import Variable

"""Function signature representation and type inference.

This module defines the ``Signature`` and ``Parameter`` classes, which represent
the typed signatures of Pixeltable functions. Signatures are inferred from Python
type hints and are used for:

- Argument validation during function calls
- Type checking (ensuring argument types match parameter types)
- Serialization/deserialization (signatures are persisted alongside computed columns)
- Batch parameter identification (parameters annotated with ``Batch[T]``)
- Overload resolution (polymorphic functions have multiple signatures)

The ``Batch`` type alias (``Batch[T] = Annotated[list[T], 'pxt-batch']``) is used
to mark parameters and return types that operate on batches of values rather than
individual scalars. This enables efficient vectorized execution.

Key classes:
    Parameter: A single typed parameter with name, kind, type, default, and batch flag.
    Signature: A complete function signature with return type, parameters, and batch info.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import typing
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

if TYPE_CHECKING:
    from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class Parameter:
    """Represents a single typed parameter of a Pixeltable function signature.

    Attributes:
        name: The parameter name as it appears in the Python function signature.
        col_type: The Pixeltable ColumnType for this parameter. None for VAR_POSITIONAL
            or VAR_KEYWORD parameters (``*args`` / ``**kwargs``).
        kind: The parameter kind (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, etc.),
            matching ``inspect.Parameter`` kinds.
        default: An optional default value, stored as a ``Literal`` expression. None means
            the parameter is required.
        is_batched: True if the parameter was annotated with ``Batch[T]``, meaning it
            receives a list of values during batched execution.
    """

    name: str
    col_type: ts.ColumnType | None  # None for variable parameters
    kind: inspect._ParameterKind
    # for some reason, this needs to precede is_batched in the dataclass definition,
    # otherwise Python complains that an argument with a default is followed by an argument without a default
    default: 'exprs.Literal' | None = None  # default value for the parameter
    is_batched: bool = False  # True if the parameter is a batched parameter (eg, Batch[dict])

    def __post_init__(self) -> None:
        from pixeltable import exprs

        if self.default is not None:
            if self.col_type is None:
                raise excs.Error(f'Cannot have a default value for variable parameter {self.name!r}')
            if not isinstance(self.default, exprs.Literal):
                raise excs.Error(f'Default value for parameter {self.name!r} is not a constant')
            if not self.col_type.is_supertype_of(self.default.col_type):
                raise excs.Error(
                    f'Default value for parameter {self.name!r} is not of type {self.col_type!r}: {self.default}'
                )

    def has_default(self) -> bool:
        """Return True if this parameter has a default value."""
        return self.default is not None

    def as_dict(self) -> dict[str, Any]:
        """Serialize this parameter to a JSON-compatible dict for persistence."""
        return {
            'name': self.name,
            'col_type': self.col_type.as_dict() if self.col_type is not None else None,
            'kind': self.kind.name,
            'is_batched': self.is_batched,
            'default': None if self.default is None else self.default.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Parameter:
        """Deserialize a Parameter from a dict produced by ``as_dict()``."""
        from pixeltable import exprs

        assert d['default'] is None or isinstance(d['default'], dict), d
        default = None if d['default'] is None else exprs.Literal.from_dict(d['default'])
        return cls(
            name=d['name'],
            col_type=ts.ColumnType.from_dict(d['col_type']) if d['col_type'] is not None else None,
            kind=getattr(inspect.Parameter, d['kind']),
            is_batched=d['is_batched'],
            default=default,
        )

    def to_py_param(self) -> inspect.Parameter:
        """Convert this Parameter to an ``inspect.Parameter`` for use in Python signature objects."""
        py_default = self.default.val if self.default is not None else inspect.Parameter.empty
        return inspect.Parameter(self.name, self.kind, default=py_default)

    def __hash__(self) -> int:
        return hash((self.name, self.col_type, self.kind, self.default, self.is_batched))


T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


class Signature:
    """Represents the typed signature of a Pixeltable function.

    A Signature captures the return type, parameter list (with types and defaults),
    and batching information for a single overload of a Pixeltable function. It is the
    central type-checking mechanism: when a function is called, argument types are validated
    against the Signature's parameter types.

    For polymorphic functions (multiple overloads), each overload has its own Signature.
    The function tries each Signature in declaration order until one matches.

    Attributes:
        return_type: The Pixeltable ColumnType returned by the function.
        is_batched: True if the return type was declared as ``Batch[T]``, indicating
            the function processes batches of inputs and returns a list of outputs.
        parameters: Ordered dict mapping parameter name to Parameter.
        parameters_by_pos: Parameters in positional order.
        constant_parameters: Parameters that are NOT batched (receive scalar values
            even during batch execution).
        batched_parameters: Parameters annotated with ``Batch[T]`` (receive lists
            during batch execution).
        required_parameters: Parameters without default values.
        system_parameters: Names of recognized internal parameters (e.g. '_runtime_ctx')
            that are excluded from the public signature.
        py_signature: A standard ``inspect.Signature`` mirror used for argument binding.

    Class attributes:
        SPECIAL_PARAM_NAMES: Reserved names ('group_by', 'order_by') that cannot be
            used as UDF parameter names because they have special semantics in aggregates.
        SYSTEM_PARAM_NAMES: Internal parameter names (e.g. '_runtime_ctx') that are
            silently stripped from the user-visible signature.
    """

    SPECIAL_PARAM_NAMES: ClassVar[list[str]] = ['group_by', 'order_by']
    SYSTEM_PARAM_NAMES: ClassVar[list[str]] = ['_runtime_ctx']

    return_type: ts.ColumnType
    is_batched: bool
    parameters: dict[str, Parameter]  # name -> Parameter
    parameters_by_pos: list[Parameter]  # ordered by position in the signature
    constant_parameters: list[Parameter]  # parameters that are not batched
    batched_parameters: list[Parameter]  # parameters that are batched
    required_parameters: list[Parameter]  # parameters that do not have a default value

    # the names of recognized system parameters in the signature; these are excluded from self.parameters
    system_parameters: list[str]

    py_signature: inspect.Signature

    def __init__(
        self,
        return_type: ts.ColumnType,
        parameters: list[Parameter],
        is_batched: bool = False,
        system_parameters: list[str] | None = None,
    ):
        assert isinstance(return_type, ts.ColumnType)
        self.return_type = return_type
        self.is_batched = is_batched
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {p.name: p for p in parameters}
        self.parameters_by_pos = parameters.copy()
        self.constant_parameters = [p for p in parameters if not p.is_batched]
        self.batched_parameters = [p for p in parameters if p.is_batched]
        self.required_parameters = [p for p in parameters if not p.has_default()]
        self.system_parameters = system_parameters if system_parameters is not None else []
        self.py_signature = inspect.Signature([p.to_py_param() for p in self.parameters_by_pos])

    def get_return_type(self) -> ts.ColumnType:
        """Return the ColumnType of this signature's return value."""
        assert isinstance(self.return_type, ts.ColumnType)
        return self.return_type

    def as_dict(self) -> dict[str, Any]:
        """Serialize this signature to a JSON-compatible dict for persistence."""
        result = {
            'return_type': self.get_return_type().as_dict(),
            'parameters': [p.as_dict() for p in self.parameters.values()],
            'is_batched': self.is_batched,
        }
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signature:
        """Deserialize a Signature from a dict produced by ``as_dict()``."""
        parameters = [Parameter.from_dict(param_dict) for param_dict in d['parameters']]
        return cls(ts.ColumnType.from_dict(d['return_type']), parameters, d['is_batched'])

    def is_consistent_with(self, other: Signature) -> bool:
        """
        Returns True if this signature is consistent with the other signature.
        S is consistent with T if we could safely replace S by T in any call where S is used. Specifically:
        (i) S.return_type is a supertype of T.return_type
        (ii) For each parameter p in S, there is a parameter q in T such that:
            - p and q have the same name and kind
            - q.col_type is a supertype of p.col_type
        (iii) For each *required* parameter q in T, there is a parameter p in S with the same name (in which
            case the kinds and types must also match, by condition (ii)).
        """
        # Check (i)
        if not self.get_return_type().is_supertype_of(other.get_return_type(), ignore_nullable=True):
            return False

        # Check (ii)
        for param_name, param in self.parameters.items():
            if param_name not in other.parameters:
                return False
            other_param = other.parameters[param_name]
            if (
                param.kind != other_param.kind
                or (param.col_type is None) != (other_param.col_type is None)  # this can happen if they are varargs
                or (
                    param.col_type is not None
                    and not other_param.col_type.is_supertype_of(param.col_type, ignore_nullable=True)
                )
            ):
                return False

        # Check (iii)
        for other_param in other.required_parameters:  # noqa: SIM110
            if other_param.name not in self.parameters:
                return False

        return True

    def validate_args(self, bound_args: dict[str, 'exprs.Expr' | None], context: str = '') -> None:
        """Validate that bound arguments are type-compatible with this signature's parameters.

        For each argument, checks that the argument's ColumnType is compatible with the
        parameter's declared ColumnType (allowing nullable arguments to non-nullable params,
        since FunctionCall.eval() handles None-skipping).

        Args:
            bound_args: Mapping from parameter names to bound Expr arguments.
            context: Optional context string for error messages (e.g. 'in function my_fn').

        Raises:
            excs.Error: If any argument type is incompatible with its parameter type.
        """
        if context:
            context = f' ({context})'

        for param_name, arg in bound_args.items():
            assert param_name in self.parameters, f'{param_name!r} not in {list(self.parameters.keys())}'
            param = self.parameters[param_name]
            is_var_param = param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            if is_var_param:
                continue
            assert param.col_type is not None

            if arg is None:
                raise excs.Error(f'Parameter {param_name!r}{context}: invalid argument')

            # Check that the argument is consistent with the expected parameter type, with the allowance that
            # non-nullable parameters can still accept nullable arguments (since in that event, FunctionCall.eval()
            # detects the Nones and skips evaluation).
            if not (
                param.col_type.is_supertype_of(arg.col_type, ignore_nullable=True)
                # TODO: this is a hack to allow JSON columns to be passed to functions that accept scalar
                # types. It's necessary to avoid littering notebooks with `apply(str)` calls or equivalent.
                # (Previously, this wasn't necessary because `is_supertype_of()` was improperly implemented.)
                # We need to think through the right way to handle this scenario.
                or (arg.col_type.is_json_type() and param.col_type.is_scalar_type())
            ):
                raise excs.Error(
                    f'Parameter {param_name!r}{context}: argument type {arg.col_type} does not'
                    f' match parameter type {param.col_type}'
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signature):
            return False
        if self.get_return_type() != other.get_return_type():
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        # ignore the parameter name
        for param, other_param in zip(self.parameters.values(), other.parameters.values()):
            if param.col_type != other_param.col_type or param.kind != other_param.kind:
                return False
        return True

    def __hash__(self) -> int:
        return hash((self.return_type, self.parameters))

    def __str__(self) -> str:
        param_strs: list[str] = []
        for p in self.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                param_strs.append(f'*{p.name}')
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                param_strs.append(f'**{p.name}')
            else:
                param_strs.append(f'{p.name}: pxt.{p.col_type}')
        return f'({", ".join(param_strs)}) -> pxt.{self.get_return_type()}'

    @classmethod
    def _infer_type(cls, annotation: type | None) -> tuple[ts.ColumnType | None, bool | None]:
        """Infer a Pixeltable ColumnType from a Python type annotation.

        Handles the ``Batch[T]`` annotation (``Annotated[list[T], 'pxt-batch']``) by
        unwrapping it to extract the inner type and setting is_batched=True.

        Args:
            annotation: A Python type annotation, or None.

        Returns:
            A tuple of (column_type, is_batched). column_type is None if the type
            cannot be mapped to a Pixeltable type.
        """
        if annotation is None:
            return (None, None)
        py_type: type | None = None
        is_batched = False
        if typing.get_origin(annotation) == typing.Annotated:
            type_args = typing.get_args(annotation)
            if len(type_args) == 2 and type_args[1] == 'pxt-batch':
                # this is our Batch
                assert typing.get_origin(type_args[0]) is list
                is_batched = True
                py_type = typing.get_args(type_args[0])[0]
        if py_type is None:
            py_type = annotation
        col_type = ts.ColumnType.from_python_type(py_type)
        return (col_type, is_batched)

    @classmethod
    def create_parameters(
        cls,
        py_fn: Callable | None = None,
        py_params: list[inspect.Parameter] | None = None,
        param_types: list[ts.ColumnType] | None = None,
        type_substitutions: dict | None = None,
        is_cls_method: bool = False,
    ) -> list[Parameter]:
        """Create a list of Parameter objects from a Python function or parameter list.

        Processes each Python parameter to determine its Pixeltable type, either from
        explicit ``param_types``, from type annotations (with optional ``type_substitutions``),
        or from the ``Batch[T]`` annotation for batched parameters.

        Skips parameters starting with '_' (reserved), system parameters, and the first
        parameter of class methods (self/cls).

        Args:
            py_fn: The Python callable to inspect. Mutually exclusive with ``py_params``.
            py_params: An explicit list of ``inspect.Parameter`` objects. Mutually exclusive
                with ``py_fn``.
            param_types: Optional explicit ColumnType list (one per parameter).
            type_substitutions: Optional mapping from annotation types to replacement types,
                used for polymorphic overloads (e.g. {T: int} substitutes generic T with int).
            is_cls_method: If True, skip the first parameter (self/cls).

        Returns:
            A list of Parameter objects with inferred types.

        Raises:
            excs.Error: If a parameter type cannot be inferred or names are reserved.
        """
        from pixeltable import exprs

        assert (py_fn is None) != (py_params is None)
        if py_fn is not None:
            sig = inspect.signature(py_fn)
            py_params = list(sig.parameters.values())
        parameters: list[Parameter] = []

        if type_substitutions is None:
            type_substitutions = {}

        for idx, param in enumerate(py_params):
            if is_cls_method and idx == 0:
                continue  # skip 'self' or 'cls' parameter
            if param.name in cls.SYSTEM_PARAM_NAMES:
                continue  # skip system parameters
            if param.name.startswith('_'):
                raise excs.Error(f"{param.name!r}: parameters starting with '_' are reserved")
            if param.name in cls.SPECIAL_PARAM_NAMES:
                raise excs.Error(f'{param.name!r} is a reserved parameter name')
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                parameters.append(Parameter(param.name, col_type=None, kind=param.kind))
                continue

            # check non-var parameters for name collisions and default value compatibility
            if param_types is not None:
                if idx >= len(param_types):
                    raise excs.Error(f'Missing type for parameter {param.name!r}')
                param_type = param_types[idx]
                is_batched = False
            else:
                # Look up the substitution for param.annotation, defaulting to param.annotation if there is none
                py_type = type_substitutions.get(param.annotation, param.annotation)
                param_type, is_batched = cls._infer_type(py_type)
                if param_type is None:
                    raise excs.Error(f'Cannot infer pixeltable type for parameter {param.name!r}')

            default = None if param.default is inspect.Parameter.empty else exprs.Expr.from_object(param.default)
            if not (default is None or isinstance(default, exprs.Literal)):
                raise excs.Error(f'Default value for parameter {param.name!r} must be a constant')

            parameters.append(
                Parameter(param.name, col_type=param_type, kind=param.kind, is_batched=is_batched, default=default)
            )

        return parameters

    @classmethod
    def create(
        cls,
        py_fn: Callable,
        param_types: list[ts.ColumnType] | None = None,
        return_type: ts.ColumnType | None = None,
        type_substitutions: dict | None = None,
        is_cls_method: bool = False,
    ) -> Signature:
        """Create a Signature by inspecting a Python callable's type annotations.

        Infers parameter types and return type from Python type hints, with optional
        explicit overrides. Handles ``Batch[T]`` annotations for batched parameters
        and return types, and filters out system parameters.

        Args:
            py_fn: The Python callable to inspect.
            param_types: Optional explicit parameter types (overrides annotation inference).
            return_type: Optional explicit return type (overrides annotation inference).
            type_substitutions: Optional mapping for polymorphic type substitution.
            is_cls_method: If True, skip the first parameter (self/cls).

        Returns:
            A fully constructed Signature.

        Raises:
            excs.Error: If parameter or return types cannot be inferred.
        """
        if type_substitutions is None:
            type_substitutions = {}

        parameters = cls.create_parameters(
            py_fn=py_fn, param_types=param_types, is_cls_method=is_cls_method, type_substitutions=type_substitutions
        )
        sig = inspect.signature(py_fn)
        if return_type is None:
            # Look up the substitution for sig.return_annotation, defaulting to return_annotation if there is none
            py_type = type_substitutions.get(sig.return_annotation, sig.return_annotation)
            return_type, return_is_batched = cls._infer_type(py_type)
            if return_type is None:
                raise excs.Error('Cannot infer pixeltable return type')
        else:
            _, return_is_batched = cls._infer_type(sig.return_annotation)
        system_params = [param_name for param_name in sig.parameters if param_name in cls.SYSTEM_PARAM_NAMES]

        return Signature(return_type, parameters, return_is_batched, system_parameters=system_params)

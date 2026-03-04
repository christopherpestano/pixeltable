"""CallableFunction -- a Pixeltable Function backed by a Python callable.

This module defines ``CallableFunction``, the most common type of Pixeltable function.
It wraps a Python callable (regular function, lambda, or async coroutine) and provides:

- **Signature inference**: Types are inferred from the callable's type annotations.
- **Batch execution**: If ``batch_size`` is set, the function processes multiple rows at
  once. Batched parameters receive lists; constant parameters receive scalars.
- **Async support**: Coroutine functions are detected and executed via the runtime event loop.
- **Serialization**: Module-level functions are serialized by path reference. Locally-defined
  functions (lambdas, notebook functions) are serialized via cloudpickle to the database.
- **Overloading**: Multiple signatures can be added via the ``.overload()`` method.

Serialization modes:
    1. Module functions: Stored by fully-qualified path (e.g. 'mymodule.my_fn').
       On load, the symbol is re-imported.
    2. Local functions: Stored by UUID in the database, with the callable pickled
       via cloudpickle and persisted as a binary blob.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Sequence
from uuid import UUID

import cloudpickle  # type: ignore[import-untyped]

import pixeltable.exceptions as excs
from pixeltable.runtime import get_runtime

from .function import Function
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import exprs


class CallableFunction(Function):
    """Pixeltable Function backed by a Python Callable.

    CallableFunctions come in two flavors:
    - **Module functions**: Defined in importable Python modules and serialized by their
      fully-qualified path (e.g. 'pixeltable.functions.openai.chat_completions'). On
      deserialization, the symbol is re-imported.
    - **Stored functions**: Defined in notebooks or lambdas, pickled via cloudpickle and
      stored in the database. Identified by UUID.

    Attributes:
        py_fns: The Python callables backing each signature (one per overload).
            For non-polymorphic functions, this is a single-element list.
        self_name: The display name of the function (used in error messages and repr).
        batch_size: If set, the function operates in batch mode. The execution engine
            will group rows into batches of this size and pass lists to batched parameters.
    """

    py_fns: list[Callable]
    self_name: str | None
    batch_size: int | None

    def __init__(
        self,
        signatures: list[Signature],
        py_fns: list[Callable],
        self_path: str | None = None,
        self_name: str | None = None,
        batch_size: int | None = None,
        is_method: bool = False,
        is_property: bool = False,
        is_deterministic: bool = True,
    ):
        assert len(signatures) > 0
        assert len(signatures) == len(py_fns)
        if self_path is None and len(signatures) > 1:
            raise excs.Error('Multiple signatures are only allowed for module UDFs (not locally defined UDFs)')
        self.py_fns = py_fns
        self.self_name = self_name
        self.batch_size = batch_size
        self.__doc__ = self.py_fns[0].__doc__
        super().__init__(
            signatures,
            self_path=self_path,
            is_method=is_method,
            is_property=is_property,
            is_deterministic=is_deterministic,
        )

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        """When resolving to a specific overload, retain only the matching callable."""
        assert len(self.py_fns) > signature_idx
        self.py_fns = [self.py_fns[signature_idx]]

    @property
    def is_batched(self) -> bool:
        """True if this function operates on batches of rows."""
        return self.batch_size is not None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.py_fn)

    def comment(self) -> str | None:
        return inspect.getdoc(self.py_fns[0])

    @property
    def py_fn(self) -> Callable:
        """The single Python callable for a non-polymorphic function."""
        assert not self.is_polymorphic
        return self.py_fns[0]

    async def aexec(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously execute this function for a single row.

        For batched functions, wraps the single-row args into singleton lists,
        calls the batch function, and extracts the single result.
        """
        assert not self.is_polymorphic
        assert self.is_async
        if self.is_batched:
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result = await self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        else:
            return await self.py_fn(*args, **kwargs)

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        """Synchronously execute this function for a single row.

        For batched functions, wraps args into singleton lists and extracts the single result.
        For async functions, runs the coroutine via the runtime event loop.
        """
        assert not self.is_polymorphic
        if self.is_batched:
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result: list[Any]
            if inspect.iscoroutinefunction(self.py_fn):
                result = get_runtime().run_coro(self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs))
            else:
                result = self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        elif inspect.iscoroutinefunction(self.py_fn):
            return get_runtime().run_coro(self.py_fn(*args, **kwargs))
        else:
            return self.py_fn(*args, **kwargs)

    async def aexec_batch(self, *args: Any, **kwargs: Any) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        assert self.is_async
        assert not self.is_polymorphic
        # Unpack the constant parameters
        constant_kwargs, batched_kwargs = self.create_batch_kwargs(kwargs)
        return await self.py_fn(*args, **constant_kwargs, **batched_kwargs)

    def exec_batch(self, args: list[Any], kwargs: dict[str, Any]) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        assert not self.is_polymorphic
        assert not self.is_async
        # Unpack the constant parameters
        constant_kwargs, batched_kwargs = self.create_batch_kwargs(kwargs)
        return self.py_fn(*args, **constant_kwargs, **batched_kwargs)

    def create_batch_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
        """Split kwargs into constant (scalar) and batched (list) kwargs.

        During batch execution, all kwargs arrive as lists. Constant parameters
        have uniform values across the batch, so we extract the first element.
        Batched parameters are passed through as-is.

        Returns:
            A tuple (constant_kwargs, batched_kwargs).
        """
        constant_param_names = [p.name for p in self.signature.constant_parameters]
        constant_kwargs = {k: v[0] for k, v in kwargs.items() if k in constant_param_names}
        batched_kwargs = {k: v for k, v in kwargs.items() if k not in constant_param_names}
        return constant_kwargs, batched_kwargs

    def get_batch_size(self, *args: Any, **kwargs: Any) -> int | None:
        return self.batch_size

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def overload(self, fn: Callable) -> CallableFunction:
        """Add an overloaded signature backed by a different Python callable.

        This allows a single UDF to accept multiple argument type combinations.
        The new callable's signature is inferred and appended to the signatures list.

        Args:
            fn: A Python callable whose type annotations define the new overload.

        Returns:
            self (for decorator chaining).

        Raises:
            excs.Error: If the function is locally defined, uses is_method/is_property,
                has already been called, or has a conditional return type.
        """
        if self.self_path is None:
            raise excs.Error('`overload` can only be used with module UDFs (not locally defined UDFs)')
        if self.is_method or self.is_property:
            raise excs.Error('`overload` cannot be used with `is_method` or `is_property`')
        if self._has_resolved_fns:
            raise excs.Error('New `overload` not allowed after the UDF has already been called')
        if self._conditional_return_type is not None:
            raise excs.Error('New `overload` not allowed after a conditional return type has been specified')
        sig = Signature.create(fn)
        self.signatures.append(sig)
        self.py_fns.append(fn)
        return self

    def _as_dict(self) -> dict:
        if self.self_path is None:
            # this is not a module function
            assert not self.is_method and not self.is_property
            from .function_registry import FunctionRegistry

            id = FunctionRegistry.get().create_stored_function(self)
            return {'id': id.hex}
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'id' in d:
            from .function_registry import FunctionRegistry

            return FunctionRegistry.get().get_stored_function(UUID(hex=d['id']))
        return super()._from_dict(d)

    def to_store(self) -> tuple[dict, bytes]:
        """Serialize this function for database storage.

        Returns a metadata dict and a cloudpickle-serialized binary blob of the callable.
        Only supported for non-polymorphic stored functions (not module-level UDFs).
        """
        assert not self.is_polymorphic  # multi-signature UDFs not allowed for stored fns
        md = {'signature': self.signature.as_dict(), 'batch_size': self.batch_size}
        return md, cloudpickle.dumps(self.py_fn)

    @classmethod
    def from_store(cls, name: str | None, md: dict, binary_obj: bytes) -> Function:
        """Reconstruct a CallableFunction from its database-stored representation."""
        py_fn = cloudpickle.loads(binary_obj)
        assert callable(py_fn)
        sig = Signature.from_dict(md['signature'])
        batch_size = md['batch_size']
        return CallableFunction([sig], [py_fn], self_name=name, batch_size=batch_size)

    def validate_call(self, bound_args: dict[str, 'exprs.Expr']) -> None:
        from pixeltable import exprs

        super().validate_call(bound_args)
        if self.is_batched:
            signature = self.signatures[0]
            for param in signature.constant_parameters:
                # Check that constant parameters map to constant arguments. It's ok for the argument to be a Variable,
                # since in that case the FunctionCall is part of an unresolved template; the check will be done again
                # when the template is fully resolved.
                if param.name in bound_args and not isinstance(bound_args[param.name], (exprs.Literal, exprs.Variable)):
                    raise ValueError(f'{self.display_name}(): parameter {param.name} must be a constant value')

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'

"""Utility helpers for symbol resolution and path validation.

This module provides low-level utilities used throughout the func package:

- ``resolve_symbol``: Resolves a dotted Python symbol path (e.g. 'pixeltable.functions.openai.chat_completions')
  into the actual Python object by progressively importing parent modules.
- ``validate_symbol_path``: Validates that a fully-qualified function path is well-formed
  (no nested/local functions, all segments are valid identifiers).
- ``get_caller_module_path``: Inspects the call stack to determine the module path of the
  caller's caller, used during function registration to auto-detect where a UDF is defined.
"""

import importlib
import inspect
from types import ModuleType

import pixeltable.exceptions as excs


def resolve_symbol(symbol_path: str) -> object | None:
    """Resolve a dotted symbol path to the Python object it refers to.

    Works by progressively shortening the module path from the right until
    a valid module is found, then traverses remaining path elements via getattr.

    For example, 'pixeltable.functions.openai.chat_completions' would first
    import 'pixeltable.functions.openai', then do getattr(module, 'chat_completions').

    Args:
        symbol_path: Fully-qualified dotted path such as 'pkg.mod.ClassName.attr'.

    Returns:
        The resolved Python object, or None if the module portion cannot be imported.

    Raises:
        AttributeError: If the module is found but the attribute path is invalid.
    """
    path_elems = symbol_path.split('.')
    module: ModuleType | None = None
    # Try progressively shorter prefixes as module paths, from longest to shortest
    i = len(path_elems) - 1
    while i > 0 and module is None:
        try:
            module = importlib.import_module('.'.join(path_elems[:i]))
        except ModuleNotFoundError:
            i -= 1
    if i == 0:
        return None  # Not resolvable
    # Traverse remaining path elements as attributes
    obj = module
    for el in path_elems[i:]:
        obj = getattr(obj, el)
    return obj


def validate_symbol_path(fn_path: str) -> None:
    """Validate that a fully-qualified function path is well-formed.

    Checks two conditions:
    1. The path does not contain '<locals>', which would indicate a nested function
       (these cannot be reliably serialized/resolved).
    2. Every segment of the dotted path is a valid Python identifier.

    Args:
        fn_path: Fully-qualified dotted path, e.g. 'mymodule.MyClass.my_fn'.

    Raises:
        excs.Error: If the path is malformed (nested function or non-identifier segment).
    """
    path_elems = fn_path.split('.')
    fn_name = path_elems[-1]
    if any(el == '<locals>' for el in path_elems):
        raise excs.Error(
            f'{fn_name}(): nested functions are not supported. Move the function to the module level or into a class.'
        )
    if any(not el.isidentifier() for el in path_elems):
        raise excs.Error(
            f'{fn_name}(): cannot resolve symbol path {fn_path}. Move the function to the module level or into a class.'
        )


def get_caller_module_path() -> str:
    """Return the module path of our caller's caller.

    Walks two frames up the call stack (past the direct caller) to find
    the module where the calling code resides. Used during decorator
    processing to determine the module where a UDF is defined.

    Returns:
        The ``__name__`` attribute of the caller's caller's module globals.
    """
    stack = inspect.stack()
    try:
        caller_frame = stack[2].frame
        module_path = caller_frame.f_globals['__name__']
    finally:
        # remove references to stack frames to avoid reference cycles
        del stack
    return module_path

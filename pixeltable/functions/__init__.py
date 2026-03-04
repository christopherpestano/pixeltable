"""
Pixeltable built-in functions module.

This package contains all built-in UDFs (User-Defined Functions) for Pixeltable, organized into
two categories:

1. **AI Provider Integrations** -- Modules that wrap external AI service APIs (e.g., anthropic,
   openai, gemini, bedrock). Each provider module registers an API client, defines async UDFs
   that call the provider's endpoints, and handles rate limiting and response parsing.

2. **Data Type Operations** -- Modules that provide operations on Pixeltable's native column
   types (e.g., string, image, audio, video, date, timestamp, json, math). These are typically
   exposed as methods/properties on column expressions (e.g., ``t.text_col.upper()``).

Global aggregate functions (sum, count, min, max, mean) and the ``map`` higher-order function
are re-exported from the ``globals`` submodule at this package level.
"""

# ruff: noqa: F401

from pixeltable.utils.code import local_public_names

# Import all provider and data-type submodules so they are accessible as
# pxt.functions.<module_name> (e.g., pxt.functions.openai, pxt.functions.image).
from . import (
    anthropic,
    audio,
    bedrock,
    date,
    deepseek,
    document,
    fabric,
    fal,
    fireworks,
    gemini,
    groq,
    huggingface,
    image,
    jina,
    json,
    llama_cpp,
    math,
    mistralai,
    net,
    ollama,
    openai,
    openrouter,
    replicate,
    reve,
    runwayml,
    string,
    timestamp,
    together,
    twelvelabs,
    uuid,
    video,
    vision,
    voyageai,
    whisper,
    whisperx,
    yolox,
)

# Re-export global aggregate/utility functions so they can be used as pxt.functions.sum(...) etc.
from .globals import count, map, max, mean, min, sum

# Build __all__ from all public names in this package plus the globals submodule.
# The 'globals' module itself is excluded from the package-level names to avoid
# shadowing Python's built-in globals().
__all__ = local_public_names(__name__, exclude=['globals']) + local_public_names(globals.__name__)


def __dir__() -> list[str]:
    return __all__

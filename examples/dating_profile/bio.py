"""LLM-driven bio generation for the dating-profile example.

This module implements Task E of the Photo-to-Dating-Profile epic
(KAN-43): given the photos selected by the upstream selection step
(Task D, KAN-47), call an LLM to compose a short first-person
dating-profile bio that actually references what is visible in the
photos.

The function exposes a narrow ``list[dict] -> str`` interface so it
can be used without a live Pixeltable database. Each selected row is
expected to look roughly like one of these shapes (the function picks
up whichever caption-like fields are present)::

    {'image': 'path/to/photo.jpg', 'caption': 'rock climbing at Yosemite'}
    {'path': '...', 'description': 'with my dog Rufus on a hike'}
    {'image': '...', 'tags': ['surfing', 'sunset', 'beach']}

The LLM call itself lives in the small ``_call_llm`` helper at the
bottom of the file. Tests patch that helper rather than going through
a real network round-trip, which keeps the unit test deterministic and
free of API credentials.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Iterable

# Word-count bounds for the generated bio. These match the acceptance
# criteria in the Jira task (KAN-48) and are also injected into the
# prompt template so the LLM is told the same numbers we later check.
MIN_WORDS = 60
MAX_WORDS = 200

# Fields on a "selected" row that we treat as caption-like signal about
# what is visible in the photo. Order matters only for output ordering.
_CAPTION_KEYS: tuple[str, ...] = ('caption', 'description', 'alt_text', 'visual_elements', 'tags', 'labels', 'objects')

_PROMPT_PATH = Path(__file__).parent / 'prompts' / 'bio.txt'

# Default model. Kept narrow on purpose: this is example code, not a
# production wrapper, and callers who want a different model can patch
# ``_call_llm`` or set the ``model`` argument on ``generate_bio``.
_DEFAULT_MODEL = 'gpt-4o-mini'


def _load_prompt_template() -> str:
    """Read the bio prompt template from disk at call time.

    Loading at call time (rather than import time) makes it easy to
    iterate on the prompt without restarting a long-running process,
    and keeps the on-disk file the single source of truth.
    """
    return _PROMPT_PATH.read_text(encoding='utf-8')


def _iter_visual_elements(selected: Iterable[dict]) -> list[str]:
    """Pull caption-like strings out of each selected row.

    We accept multiple key names because the upstream selection step
    (Task D) is still in flight and the exact schema may shift. List
    values (e.g. ``tags=['surfing', 'beach']``) are flattened.
    Duplicate phrases are dropped while preserving first-seen order.
    """
    seen: set[str] = set()
    elements: list[str] = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        for key in _CAPTION_KEYS:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                candidates = [value]
            elif isinstance(value, (list, tuple, set)):
                candidates = [str(v) for v in value]
            else:
                candidates = [str(value)]
            for candidate in candidates:
                cleaned = candidate.strip()
                if not cleaned:
                    continue
                key_lower = cleaned.lower()
                if key_lower in seen:
                    continue
                seen.add(key_lower)
                elements.append(cleaned)
    return elements


def _format_visual_elements(elements: list[str]) -> str:
    if not elements:
        return '(no captions were available for the selected photos)'
    return '\n'.join(f'- {e}' for e in elements)


def _format_user_facts(user_facts: dict[str, Any] | None) -> str:
    if not user_facts:
        return '(none provided)'
    lines: list[str] = []
    for key, value in user_facts.items():
        if value is None or value == '':
            continue
        if isinstance(value, (list, tuple, set)):
            rendered = ', '.join(str(v) for v in value)
        else:
            rendered = str(value)
        lines.append(f'- {key}: {rendered}')
    return '\n'.join(lines) if lines else '(none provided)'


def build_prompt(selected: list[dict], user_facts: dict[str, Any] | None = None) -> str:
    """Render the bio prompt from the template plus the inputs.

    Exposed publicly (no leading underscore) so callers can inspect or
    log the exact prompt that was sent to the LLM - useful when
    debugging unexpected bios.
    """
    template = Template(_load_prompt_template())
    return template.safe_substitute(
        visual_elements=_format_visual_elements(_iter_visual_elements(selected)),
        user_facts=_format_user_facts(user_facts),
        min_words=str(MIN_WORDS),
        max_words=str(MAX_WORDS),
    )


def _call_llm(prompt: str, *, model: str = _DEFAULT_MODEL) -> str:
    """Issue a completion against the configured LLM backend.

    This is intentionally a tiny seam: tests patch this function out
    rather than mocking the underlying SDK. The default implementation
    uses the OpenAI Python SDK if it is installed and configured;
    callers who want a different backend can monkeypatch this function
    or wrap ``generate_bio`` themselves.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            'Bio generation requires the `openai` package by default. '
            'Either `pip install openai` and set OPENAI_API_KEY, or '
            'monkeypatch `examples.dating_profile.bio._call_llm` to use a '
            'different LLM backend.'
        ) from e

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': (
                    "You write authentic, first-person dating-profile bios that reflect the user's actual photos."
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content or ''


def generate_bio(selected: list[dict], user_facts: dict[str, Any] | None = None) -> str:
    """Generate a dating-profile bio from the selected photos.

    Args:
        selected: The rows produced by the photo-selection step (Task D,
            KAN-47). Each row is a ``dict`` describing one of the
            user's chosen photos. The function looks for caption-like
            fields (``caption``, ``description``, ``tags``, ...) and
            feeds them to the LLM so the bio can reference what is
            actually visible.
        user_facts: Optional user-provided facts such as
            ``{'age': 31, 'city': 'Oakland', 'interests': [...]}``.
            Only fields the caller provides are passed through; the
            function never invents facts.

    Returns:
        A first-person bio, between ``MIN_WORDS`` and ``MAX_WORDS``
        words, that references at least two distinct visual elements
        drawn from ``selected``.

    Raises:
        ValueError: If ``selected`` is empty.

    Example:
        ```python
        from examples.dating_profile.bio import generate_bio

        selected = [
            {'image': 'a.jpg', 'caption': 'rock climbing at Yosemite'},
            {'image': 'b.jpg', 'caption': 'with my dog on a trail'},
        ]
        bio = generate_bio(selected, user_facts={'age': 31, 'city': 'Oakland'})
        print(bio)
        ```
    """
    if not selected:
        raise ValueError('generate_bio: `selected` must contain at least one photo row.')

    prompt = build_prompt(selected, user_facts)
    bio = _call_llm(prompt)
    return bio.strip()

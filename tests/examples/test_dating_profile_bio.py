"""Unit tests for ``examples.dating_profile.bio.generate_bio``.

The LLM call is patched out: we want these tests to run offline, in CI,
without API credentials. Each test supplies fake "selected" rows with
known captions, mocks the LLM to return a predetermined bio, and then
asserts the contract of ``generate_bio`` -- that the returned text
references the visual elements drawn from the photos and falls inside
the documented length bounds.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from examples.dating_profile import bio as bio_module
from examples.dating_profile.bio import MAX_WORDS, MIN_WORDS, generate_bio


def _fake_bio_referencing_climbing_and_dog() -> str:
    """A hand-rolled bio inside the word bounds that mentions both elements.

    Word count is ~110, comfortably inside [60, 200].
    """
    return (
        "Hi, I'm someone who genuinely likes being outside. On a good "
        "weekend you'll catch me rock climbing at the local crag, chalk "
        "on my hands and the kind of grin that doesn't come from a "
        "desk. The rest of the time I'm usually with my dog Rufus, "
        'taking him on long ambling walks or trying to teach him a new '
        '(questionable) trick. During the week I work in software, but '
        'I try not to be the person whose laptop comes everywhere. I '
        'cook a lot, read more than I probably should, and have strong '
        "opinions about coffee. If you're up for a hike, a bouldering "
        'gym, or just trading dog photos, say hi.'
    )


def _word_count(text: str) -> int:
    return len(text.split())


class TestGenerateBio:
    def test_returns_text_with_visual_keywords_within_bounds(self) -> None:
        """Main acceptance-criteria test.

        Mocks the LLM, supplies fake selected rows with known captions,
        asserts the returned text references at least the two expected
        visual elements ("rock climbing" and "dog") and falls within
        [MIN_WORDS, MAX_WORDS] words.
        """
        selected = [
            {'image': 'photo1.jpg', 'caption': 'rock climbing at Yosemite, chalk on hands'},
            {'image': 'photo2.jpg', 'caption': 'with my dog Rufus on a hike'},
            {'image': 'photo3.jpg', 'caption': 'cooking pasta in a kitchen'},
        ]
        user_facts = {'age': 31, 'city': 'Oakland', 'interests': ['climbing', 'cooking']}

        fake_bio = _fake_bio_referencing_climbing_and_dog()
        with patch.object(bio_module, '_call_llm', return_value=fake_bio) as mock_llm:
            result = generate_bio(selected, user_facts=user_facts)

        # The LLM seam was actually exercised.
        mock_llm.assert_called_once()

        # Returned text references the expected visual elements.
        lowered = result.lower()
        assert 'rock climbing' in lowered, f'expected "rock climbing" in bio, got: {result!r}'
        assert 'dog' in lowered, f'expected "dog" in bio, got: {result!r}'

        # And falls inside the documented length bounds.
        word_count = _word_count(result)
        assert MIN_WORDS <= word_count <= MAX_WORDS, (
            f'bio word count {word_count} outside [{MIN_WORDS}, {MAX_WORDS}]; bio: {result!r}'
        )

    def test_visual_elements_are_threaded_into_the_prompt(self) -> None:
        """The captions on selected rows must reach the LLM prompt.

        Without this, the LLM has nothing to reference and the
        "at least 2 distinct visual elements" criterion is unverifiable.
        """
        selected: list[dict] = [
            {'image': 'a.jpg', 'caption': 'surfing in Hawaii at sunset'},
            {'image': 'b.jpg', 'tags': ['playing guitar', 'campfire']},
        ]

        captured: dict[str, str] = {}

        def fake_llm(prompt: str, **_kwargs: object) -> str:
            captured['prompt'] = prompt
            # Return something inside the length bounds so generate_bio
            # itself doesn't choke. We're asserting against the prompt,
            # not the bio.
            return _fake_bio_referencing_climbing_and_dog()

        with patch.object(bio_module, '_call_llm', side_effect=fake_llm):
            generate_bio(selected)

        prompt = captured['prompt']
        assert 'surfing in Hawaii at sunset' in prompt
        assert 'playing guitar' in prompt
        assert 'campfire' in prompt

    def test_user_facts_are_threaded_into_the_prompt(self) -> None:
        selected = [{'caption': 'kayaking on a river'}]

        captured: dict[str, str] = {}

        def fake_llm(prompt: str, **_kwargs: object) -> str:
            captured['prompt'] = prompt
            return _fake_bio_referencing_climbing_and_dog()

        with patch.object(bio_module, '_call_llm', side_effect=fake_llm):
            generate_bio(selected, user_facts={'age': 29, 'city': 'Portland'})

        prompt = captured['prompt']
        assert '29' in prompt
        assert 'Portland' in prompt

    def test_user_facts_optional(self) -> None:
        selected = [{'caption': 'reading a book in a hammock'}]

        with patch.object(bio_module, '_call_llm', return_value=_fake_bio_referencing_climbing_and_dog()):
            result = generate_bio(selected)  # no user_facts at all

        assert MIN_WORDS <= _word_count(result) <= MAX_WORDS

    def test_empty_selection_raises(self) -> None:
        with pytest.raises(ValueError, match='at least one photo'):
            generate_bio([])

    def test_result_is_stripped(self) -> None:
        """LLM output with stray leading/trailing whitespace is trimmed."""
        padded = '\n\n   ' + _fake_bio_referencing_climbing_and_dog() + '   \n'
        selected = [{'caption': 'rock climbing'}, {'caption': 'with my dog'}]

        with patch.object(bio_module, '_call_llm', return_value=padded):
            result = generate_bio(selected)

        assert not result.startswith((' ', '\n'))
        assert not result.endswith((' ', '\n'))


class TestPromptTemplate:
    def test_prompt_template_loaded_from_disk(self) -> None:
        """The template lives in a separate file and is loaded at runtime.

        This guards against accidentally inlining the prompt into Python
        source, which would violate the acceptance criterion that the
        template be in ``examples/dating_profile/prompts/bio.txt``.
        """
        raw = bio_module._load_prompt_template()
        # The template uses string.Template placeholders for the dynamic
        # bits.
        assert '$visual_elements' in raw
        assert '$user_facts' in raw
        assert '$min_words' in raw
        assert '$max_words' in raw

    def test_prompt_template_path_exists(self) -> None:
        assert bio_module._PROMPT_PATH.exists(), f'prompt template missing at {bio_module._PROMPT_PATH}'
        assert bio_module._PROMPT_PATH.name == 'bio.txt'
        assert bio_module._PROMPT_PATH.parent.name == 'prompts'

    def test_build_prompt_substitutes_placeholders(self) -> None:
        prompt = bio_module.build_prompt([{'caption': 'snowboarding in Tahoe'}], user_facts={'age': 27})
        # Placeholders are gone, content is in.
        assert '$visual_elements' not in prompt
        assert '$user_facts' not in prompt
        assert '$min_words' not in prompt
        assert '$max_words' not in prompt
        assert 'snowboarding in Tahoe' in prompt
        assert '27' in prompt
        assert str(MIN_WORDS) in prompt
        assert str(MAX_WORDS) in prompt

    def test_build_prompt_handles_missing_captions(self) -> None:
        """Rows without caption-like fields don't crash prompt building."""
        prompt = bio_module.build_prompt([{'image': 'a.jpg'}, {'image': 'b.jpg'}])
        # The fallback "no captions available" string lands in the prompt.
        assert 'no captions' in prompt.lower()

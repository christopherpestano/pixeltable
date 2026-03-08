"""Tests for Ray scheduler — all tests run without a Ray cluster (local fallback path)."""

import os

import pytest

import pixeltable as pxt
from pixeltable.exec.expr_eval.schedulers import RayScheduler
from pixeltable.func import Batch

from .utils import get_image_files, skip_test_if_not_installed, validate_update_status

IN_CI = os.environ.get('CI') is not None


class TestRaySchedulerMatches:
    def test_matches_ray(self) -> None:
        assert RayScheduler.matches('ray')

    def test_matches_ray_qualified(self) -> None:
        assert RayScheduler.matches('ray:gpu')
        assert RayScheduler.matches('ray:my-pool')

    def test_no_match_other(self) -> None:
        assert not RayScheduler.matches('request-rate:openai')
        assert not RayScheduler.matches('rate-limits:openai')
        assert not RayScheduler.matches('rayon')


@pxt.udf(resource_pool='ray')
def _double_it(x: int) -> int:
    return x * 2


@pxt.udf(resource_pool='ray', batch_size=4)
def _double_batch(x: Batch[int]) -> Batch[int]:
    return [v * 2 for v in x]


@pxt.udf
def _triple_it(x: int) -> int:
    return x * 3


@pxt.udf
def _multiply(x: int, factor: int = 2) -> int:
    return x * factor


class TestRayLocalFallback:
    """Tests that UDFs with resource_pool='ray' execute correctly via local fallback."""

    def test_local_fallback_scalar(self, init_env) -> None:  # type: ignore[no-untyped-def]
        t = pxt.create_table('test_ray_scalar', {'val': pxt.Int})
        t.add_computed_column(doubled=_double_it(t.val))
        t.insert([{'val': i} for i in range(5)])
        results = t.select(t.val, t.doubled).order_by(t.val).collect()
        for row in results:
            assert row['doubled'] == row['val'] * 2

    def test_local_fallback_batched(self, init_env) -> None:  # type: ignore[no-untyped-def]
        t = pxt.create_table('test_ray_batched', {'val': pxt.Int})
        t.add_computed_column(doubled=_double_batch(t.val))
        t.insert([{'val': i} for i in range(8)])
        results = t.select(t.val, t.doubled).order_by(t.val).collect()
        for row in results:
            assert row['doubled'] == row['val'] * 2


class TestRayConfig:
    def test_ray_config_env_var(self, init_env) -> None:  # type: ignore[no-untyped-def]
        """Verify RAY_ADDRESS env var is picked up by Config."""
        from pixeltable.config import Config

        old = os.environ.get('RAY_ADDRESS')
        try:
            os.environ['RAY_ADDRESS'] = 'ray://test-host:10001'
            # Re-init config to pick up the env var
            Config.init({}, reinit=True)
            config = Config.get()
            assert config.get_string_value('address', section='ray') == 'ray://test-host:10001'
        finally:
            if old is not None:
                os.environ['RAY_ADDRESS'] = old
            else:
                os.environ.pop('RAY_ADDRESS', None)
            # Re-init config to restore original state
            Config.init({}, reinit=True)


class TestRayUsingResourcePool:
    """Tests that .using(resource_pool='ray') overrides the resource pool on existing UDFs."""

    def test_using_resource_pool_on_plain_udf(self, init_env) -> None:  # type: ignore[no-untyped-def]
        """A UDF without a resource pool can be overridden to use 'ray' via .using()."""
        t = pxt.create_table('test_using_rp', {'val': pxt.Int})
        ray_triple = _triple_it.using(resource_pool='ray')
        t.add_computed_column(tripled=ray_triple(t.val))
        t.insert([{'val': i} for i in range(5)])
        results = t.select(t.val, t.tripled).order_by(t.val).collect()
        for row in results:
            assert row['tripled'] == row['val'] * 3

    def test_using_resource_pool_with_params(self, init_env) -> None:  # type: ignore[no-untyped-def]
        """resource_pool can be combined with other .using() parameter bindings."""
        t = pxt.create_table('test_using_rp_params', {'val': pxt.Int})
        ray_times5 = _multiply.using(factor=5, resource_pool='ray')
        t.add_computed_column(result=ray_times5(t.val))
        t.insert([{'val': i} for i in range(4)])
        results = t.select(t.val, t.result).order_by(t.val).collect()
        for row in results:
            assert row['result'] == row['val'] * 5


class TestRayHuggingFace:
    """Tests for HuggingFace UDFs offloaded via Ray (local fallback).

    These tests download large models and are expensive to run.
    They verify that HuggingFace image/video generation UDFs work correctly
    when routed through the RayScheduler's local fallback path.
    """

    @pytest.mark.skipif(IN_CI, reason='Large model download; skipped in CI')
    @pytest.mark.expensive
    def test_text_to_image_ray(self, init_env) -> None:  # type: ignore[no-untyped-def]
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        from pixeltable.functions.huggingface import text_to_image

        t = pxt.create_table('test_hf_t2i_ray', {'prompt': pxt.String})
        t.add_computed_column(
            image=text_to_image.using(resource_pool='ray')(
                t.prompt,
                model_id='stabilityai/stable-diffusion-xl-base-1.0',
                height=512,
                width=512,
                seed=42,
                model_kwargs={
                    'num_inference_steps': 15,
                    'guidance_scale': 7.5,
                    'negative_prompt': 'blurry, low quality, distorted, deformed',
                },
            )
        )
        validate_update_status(
            t.insert(prompt='a red circle on a white background, minimalist, clean vector art'), expected_rows=1
        )
        result = t.select(t.image).collect()[0]
        assert result['image'] is not None

    @pytest.mark.skipif(IN_CI, reason='Large model download; skipped in CI')
    @pytest.mark.expensive
    def test_image_to_image_ray(self, init_env) -> None:  # type: ignore[no-untyped-def]
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        from pixeltable.functions.huggingface import image_to_image

        image_files = get_image_files()
        t = pxt.create_table('test_hf_i2i_ray', {'img': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(
            modified=image_to_image.using(resource_pool='ray')(
                t.img,
                t.prompt,
                model_id='stabilityai/stable-diffusion-xl-base-1.0',
                model_kwargs={
                    'strength': 0.6,
                    'num_inference_steps': 15,
                    'guidance_scale': 7.5,
                    'negative_prompt': 'blurry, low quality, distorted, artifacts',
                },
            )
        )
        validate_update_status(
            t.insert(img=image_files[0], prompt='oil painting style, vibrant colors, visible brushstrokes'),
            expected_rows=1,
        )
        result = t.select(t.modified).collect()[0]
        assert result['modified'] is not None

    @pytest.mark.skipif(IN_CI, reason='Large model download; skipped in CI')
    @pytest.mark.expensive
    def test_image_to_video_ray(self, init_env) -> None:  # type: ignore[no-untyped-def]
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        from pixeltable.functions.huggingface import image_to_video

        image_files = get_image_files()
        t = pxt.create_table('test_hf_i2v_ray', {'img': pxt.Image})
        t.add_computed_column(
            video=image_to_video.using(resource_pool='ray')(
                t.img,
                model_id='stabilityai/stable-video-diffusion-img2vid-xt',
                num_frames=2,
                model_kwargs={'num_inference_steps': 3},
            )
        )
        validate_update_status(t.insert(img=image_files[0]), expected_rows=1)
        result = t.select(t.video).collect()[0]
        assert result['video'] is not None

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke tests for both UniRT backends.

Run from the repo root after building and installing (see README):

    python3 -m pytest tests/python -v
"""

from conftest import ask_capital, model_path
import pytest

from unirt._ffi._api import UniRTError


class TestCore:
    def test_both_runtimes_load(self, sdk):
        runtimes = sdk.get_runtime_list()
        assert 'llama_cpp' in runtimes
        if 'mlx' not in runtimes:
            import pytest
            pytest.skip('MLX plugin is not built in this configuration')

    def test_device_lists(self, sdk):
        mlx_devices = (
            sdk.get_compute_unit_list('mlx')
            if 'mlx' in sdk.get_runtime_list() else []
        )
        assert len(mlx_devices) <= 1
        if mlx_devices:
            assert mlx_devices[0][0] == 'mlx0'
        llama_devices = sdk.get_compute_unit_list('llama_cpp')
        assert any(dev_id.startswith('MTL') or dev_id == 'CPU' for dev_id, _ in llama_devices)


class TestLlamaCpp:
    def test_generate_capital(self, llama_model):
        assert 'Paris' in ask_capital(llama_model)

    def test_runtime_stats(self, llama_model):
        s = llama_model.runtime_stats()
        assert s['model_bytes'] > 100_000_000  # Q8_0 weights ≈ 138MB
        assert s['process_rss_bytes'] > 0

    def test_default_generation_replaces_old_kv_state(self, llama_model):
        prompt = llama_model._apply_chat_template(
            [{'role': 'user', 'content': 'Reply with exactly three short words.'}],
            True,
            False,
            None,
        )
        first = llama_model.generate(prompt, max_new_tokens=12).text
        # No reset here: n_past=0 must trim the prior KV before decoding the
        # full prompt again.
        second = llama_model.generate(prompt, max_new_tokens=12).text
        assert first == second

    def test_stop_sequence_never_leaks_to_stream(self, llama_model):
        prompt = llama_model._apply_chat_template(
            [{'role': 'user', 'content': 'What is the capital of France?'}],
            True,
            False,
            None,
        )
        baseline = llama_model.generate(prompt, max_new_tokens=24).text
        assert len(baseline) >= 4
        stop = baseline[2:4]

        streamer = llama_model.generate(prompt, max_new_tokens=24, stop=[stop], stream=True)
        streamed = ''.join(streamer)
        assert streamer.output is not None
        assert streamed == streamer.output.text
        assert stop not in streamed
        assert streamer.output.profile.stop_reason == 'stop_sequence'

    def test_invalid_n_past_is_rejected_without_crashing(self, llama_model):
        with pytest.raises(UniRTError):
            llama_model.generate('hello', max_new_tokens=1, n_past=2_000_000_000)

    def test_runtime_cannot_unload_live_plugin_vtables(self, sdk, llama_model):
        assert sdk.load_library().unirt_deinit() == -1010


class TestMlx:
    def test_generate_capital(self, mlx_model):
        assert 'Paris' in ask_capital(mlx_model)

    def test_chat_template_is_chatml(self, mlx_model):
        prompt = mlx_model._apply_chat_template(
            [{'role': 'user', 'content': 'hi'}], True, False, None)
        assert prompt == '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n'

    def test_greedy_is_deterministic(self, mlx_model):
        assert ask_capital(mlx_model) == ask_capital(mlx_model)

    def test_sampling_varies_with_seed(self, mlx_model):
        a = ask_capital(mlx_model, temperature=1.2, top_p=0.9, seed=1)
        b = ask_capital(mlx_model, temperature=1.2, top_p=0.9, seed=2)
        assert a != b

    def test_streaming_matches_blocking(self, mlx_model):
        prompt = mlx_model._apply_chat_template(
            [{'role': 'user', 'content': 'Count from 1 to 5.'}], True, False, None)
        streamer = mlx_model.generate(prompt, max_new_tokens=32, stream=True)
        streamed = ''.join(streamer)
        assert streamed == streamer.output.text
        mlx_model.reset()

    def test_streaming_multibyte_not_mangled(self, mlx_model):
        # CJK output is routinely split mid-character across token pieces;
        # the streamer must reassemble it instead of emitting U+FFFD.
        prompt = mlx_model._apply_chat_template(
            [{'role': 'user', 'content': '請用繁體中文說「你好，世界」。'}], True, False, None)
        streamer = mlx_model.generate(prompt, max_new_tokens=48, stream=True)
        streamed = ''.join(streamer)
        assert streamed == streamer.output.text
        assert '�' not in streamed
        mlx_model.reset()

    def test_runtime_stats(self, mlx_model):
        s = mlx_model.runtime_stats()
        assert s['model_bytes'] > 200_000_000  # bf16 weights ≈ 269MB
        assert s['device_peak_bytes'] > 0
        assert s['device_name'] and 'MLX' in s['device_name']

    def test_quantized_model(self, sdk):
        from unirt.auto import AutoModelForCausalLM

        from conftest import require_mlx

        require_mlx(sdk)

        m = AutoModelForCausalLM.from_pretrained(model_path('safetensors_8bit'), device_map='mlx')
        try:
            assert 'Paris' in ask_capital(m)
            # 8-bit packed weights must be materially smaller than bf16
            assert m.runtime_stats()['model_bytes'] < 230_000_000
        finally:
            m.close()

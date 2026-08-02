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


class TestBosVocabulary:
    """The BOS path, which SmolLM2 (add_bos_token=false) cannot exercise.

    Tokenizing the prompt with add_special keyed off "is the KV cache empty"
    rather than off n_past made every turn but the first arrive without BOS.
    On Gemma that emptied the reply outright, and because history_[0] was then
    the cached BOS against a prompt starting at the first real token, the
    prefix match failed at position 0 and re-prefilled the whole transcript
    every turn. Both symptoms are invisible on a vocabulary that adds no BOS,
    which is why the suite needs a model that does.
    """

    @pytest.fixture(scope='class')
    def bos_model(self, sdk):
        from unirt.auto import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path('gguf_bos'), device_map='llama_cpp'
        )
        yield model
        model.close()

    @staticmethod
    def _ask(model):
        return model._apply_chat_template(
            [{'role': 'user', 'content': 'Name one color.'}], True, False, None
        )

    def test_repeating_a_prompt_repeats_the_answer(self, bos_model):
        prompt = self._ask(bos_model)
        first = bos_model.generate(prompt, max_new_tokens=8, temperature=0.0)
        # Deliberately no reset: the second call must rebuild the identical
        # token sequence, BOS included, and so produce the identical output.
        second = bos_model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert first.text
        assert second.text == first.text
        assert second.profile.prompt_tokens == first.profile.prompt_tokens

    def test_a_cold_prompt_and_a_warm_one_tokenize_alike(self, bos_model):
        prompt = self._ask(bos_model)
        bos_model.reset()
        cold = bos_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        warm = bos_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert warm.profile.prompt_tokens == cold.profile.prompt_tokens


class TestMlx:
    def test_generate_capital(self, mlx_model):
        assert 'Paris' in ask_capital(mlx_model)

    def test_context_overflow_evicts_instead_of_stopping(self, sdk):
        from conftest import require_mlx

        require_mlx(sdk)
        from unirt.auto import AutoModelForCausalLM

        # A context this small guarantees the request below overflows it,
        # forcing the KV-cache eviction path (sdk/plugins/mlx/src/plugin.cpp
        # shift_context()) instead of a hard "context_length" stop.
        m = AutoModelForCausalLM.from_pretrained(model_path('safetensors'), device_map='mlx', n_ctx=48)
        try:
            prompt = m._apply_chat_template(
                [{'role': 'user', 'content': 'Count from one to one hundred, one number per line.'}],
                True, False, None)
            out = m.generate(prompt, max_new_tokens=120)
            assert out.text
            assert out.profile.stop_reason != 'context_length'
            m.reset()
        finally:
            m.close()

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


class TestSharedWeights:
    """A second handle on one model is a second KV cache, not a second model.

    This is what makes `--slots N` affordable: N contexts over one set of
    parameters. Both text backends load through a process-wide cache keyed on
    the model, and each says so when it hits.
    """

    @pytest.mark.parametrize(
        'backend,kind',
        [('llama_cpp', 'gguf'), ('mlx', 'safetensors')],
    )
    def test_a_second_handle_reuses_the_first_ones_weights(self, sdk, backend, kind, caplog):
        import logging

        from unirt.auto import AutoModelForCausalLM

        if backend == 'mlx':
            from conftest import require_mlx

            require_mlx(sdk)
        path = model_path(kind)

        first = AutoModelForCausalLM.from_pretrained(path, device_map=backend)
        try:
            with caplog.at_level(logging.DEBUG, logger='unirt'):
                second = AutoModelForCausalLM.from_pretrained(path, device_map=backend)
            try:
                assert any(
                    'reusing already-loaded weights' in record.message
                    for record in caplog.records
                ), 'the second handle loaded its own copy of the weights'
                # Sharing must not have coupled the two contexts: each keeps
                # its own KV cache, so what one generated is not in the other.
                assert 'Paris' in ask_capital(second)
                assert second.runtime_stats()['model_bytes'] == \
                    first.runtime_stats()['model_bytes']
            finally:
                second.close()
            # The surviving handle still owns working weights after the other
            # one released its reference.
            assert 'Paris' in ask_capital(first)
        finally:
            first.close()

    def test_weights_are_released_once_every_handle_is_closed(self, sdk, caplog, tmp_path):
        """The cache holds a weak reference, so closing everything frees it."""
        import logging
        import shutil

        from unirt.auto import AutoModelForCausalLM

        # A copy of its own, so the result does not depend on whether some
        # other test in this session still holds the shared fixture open.
        path = str(tmp_path / 'copy.gguf')
        shutil.copyfile(model_path('gguf'), path)

        first = AutoModelForCausalLM.from_pretrained(path, device_map='llama_cpp')
        first.close()
        with caplog.at_level(logging.DEBUG, logger='unirt'):
            second = AutoModelForCausalLM.from_pretrained(path, device_map='llama_cpp')
        try:
            assert not any(
                'reusing already-loaded weights' in record.message
                for record in caplog.records
            ), 'a closed handle left its weights resident'
        finally:
            second.close()


class TestSpeculativeDecoding:
    """A draft model changes how tokens are produced, never which ones.

    The target verifies every proposal against its own logits and keeps only
    what it agrees with, so speculation is an optimisation with an exact
    correctness criterion: same prompt, same settings, same text.
    """

    @staticmethod
    def _prompt(model):
        return model._apply_chat_template(
            [{'role': 'user', 'content': 'Write two sentences about the sea.'}],
            True, False, None,
        )

    @pytest.fixture(scope='class')
    def self_drafting(self, sdk):
        """The model drafting for itself.

        Not a speedup -- drafting costs exactly what it saves -- but the
        sharpest correctness test available with one model: every proposal
        should be accepted, which is precisely the path where a mistake in the
        verification arithmetic or the KV rollback has nowhere to hide.
        """
        from unirt.auto import AutoModelForCausalLM

        path = model_path('gguf')
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map='llama_cpp', n_ctx=1024, draft_model=path
        )
        yield model
        model.close()

    def test_speculation_does_not_change_the_answer(self, llama_model, self_drafting):
        llama_model.reset()
        prompt = self._prompt(llama_model)
        plain = llama_model.generate(prompt, max_new_tokens=48, temperature=0.0)
        self_drafting.reset()
        speculated = self_drafting.generate(prompt, max_new_tokens=48, temperature=0.0)
        assert speculated.text == plain.text
        assert speculated.profile.generated_tokens == plain.profile.generated_tokens

    def test_stop_sequences_still_stop(self, self_drafting):
        """A stop can land in the middle of a verified batch, which is the one
        place the token loop has to stop early on tokens already in the KV."""
        self_drafting.reset()
        out = self_drafting.generate(
            self._prompt(self_drafting), max_new_tokens=48, temperature=0.0, stop=['sea']
        )
        assert out.profile.stop_reason == 'stop_sequence'
        assert 'sea' not in out.text

    def test_the_handle_still_works_for_the_next_turn(self, self_drafting):
        """Rolling back rejected tokens must leave the KV and the transcript
        agreeing, or the next turn reuses a prefix that is not there."""
        prompt = self._prompt(self_drafting)
        self_drafting.reset()
        first = self_drafting.generate(prompt, max_new_tokens=24, temperature=0.0)
        second = self_drafting.generate(prompt, max_new_tokens=24, temperature=0.0)
        assert first.text == second.text

    def test_speculation_can_be_turned_off_per_request(self, self_drafting):
        self_drafting.reset()
        prompt = self._prompt(self_drafting)
        with_draft = self_drafting.generate(prompt, max_new_tokens=32, temperature=0.0)
        self_drafting.reset()
        without = self_drafting.generate(
            prompt, max_new_tokens=32, temperature=0.0, n_draft=-1
        )
        assert with_draft.text == without.text

    def test_logprobs_and_grammars_still_work_with_a_draft_attached(self, self_drafting):
        """Both fall back to plain decoding -- the results have to be the same
        as if no draft had been loaded at all."""
        import json

        self_drafting.reset()
        scored = self_drafting.generate(
            self._prompt(self_drafting), max_new_tokens=8, temperature=0.0, logprobs=2
        )
        assert scored.logprobs and len(scored.logprobs) == 8
        self_drafting.reset()
        constrained = self_drafting.generate(
            self._prompt(self_drafting), max_new_tokens=48, temperature=0.0,
            json_schema={'type': 'object', 'properties': {'a': {'type': 'string'}},
                         'required': ['a']},
        )
        assert 'a' in json.loads(constrained.text)

    def test_a_draft_model_with_a_different_vocabulary_is_refused(self, sdk):
        """Proposals are token ids. Against a different vocabulary they are
        different words, so this cannot be allowed to load and silently
        produce nonsense the target then has to reject every time."""
        from unirt.auto import AutoModelForCausalLM
        from unirt._ffi._api import UniRTError

        with pytest.raises(UniRTError):
            AutoModelForCausalLM.from_pretrained(
                model_path('gguf'), device_map='llama_cpp', n_ctx=512,
                draft_model=model_path('gguf_bos'),
            ).close()

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'bindings', 'python'))

MODELS = {
    'gguf': os.path.join(REPO_ROOT, 'models', 'SmolLM2-135M-Instruct-Q8_0.gguf'),
    'safetensors': os.path.join(REPO_ROOT, 'models', 'SmolLM2-135M-Instruct'),
    'safetensors_8bit': os.path.join(REPO_ROOT, 'models', 'SmolLM2-135M-Instruct-8bit'),
}


def model_path(kind: str) -> str:
    path = MODELS[kind]
    if not os.path.exists(path):
        pytest.skip(f'test model not present: {path} (see README for download commands)')
    return path


@pytest.fixture(scope='session')
def sdk():
    from unirt._ffi import _api

    _api.init()
    yield _api
    _api.deinit()


def _load(path: str, backend: str):
    from unirt.auto import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, device_map=backend)


@pytest.fixture(scope='session')
def llama_model(sdk):
    m = _load(model_path('gguf'), 'llama_cpp')
    yield m
    m.close()


@pytest.fixture(scope='session')
def mlx_model(sdk):
    if not sdk.get_compute_unit_list('mlx'):
        pytest.skip('MLX plugin is present but no usable Metal device is available')
    m = _load(model_path('safetensors'), 'mlx')
    yield m
    m.close()


CAPITAL_MESSAGES = [{'role': 'user', 'content': 'What is the capital of France? Answer in one sentence.'}]


def ask_capital(model, **gen_kwargs) -> str:
    prompt = model._apply_chat_template(CAPITAL_MESSAGES, True, False, None)
    out = model.generate(prompt, max_new_tokens=48, **gen_kwargs)
    model.reset()
    return out.text

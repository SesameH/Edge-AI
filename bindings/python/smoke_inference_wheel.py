#!/usr/bin/env python3
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate text with an installed wheel: proof the shipped binary runs.

``smoke_installed_wheel.py`` only proves the plugin can be dlopen'd. That
catches a broken packaging layout but not a runtime that faults on the first
forward pass, which is the failure a cross-compiled wheel is most likely to
have. This one loads a real GGUF and decodes, then checks the two properties
the test suite treats as load-bearing: greedy decoding is deterministic, and a
schema-constrained reply parses.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCHEMA = {
    'type': 'object',
    'properties': {'colour': {'type': 'string'}},
    'required': ['colour'],
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: smoke_inference_wheel.py <model.gguf>')
    model_path = Path(sys.argv[1]).resolve()
    if not model_path.is_file():
        raise SystemExit(f'no such model file: {model_path}')

    # Keep the source tree off sys.path so this exercises the installed
    # package rather than the checkout it happens to live in.
    source_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != source_dir
    ]
    os.environ.pop('UNIRT_LIB_PATH', None)
    os.environ.pop('UNIRT_PLUGIN_PATH', None)

    from unirt import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map='llama_cpp', n_ctx=512
    )
    try:
        prompt = model._apply_chat_template(
            [{'role': 'user', 'content': 'Name one colour.'}], True, False, None
        )
        first = model.generate(prompt, max_new_tokens=16, temperature=0.0)
        if not first.text.strip():
            raise RuntimeError('generation produced no text')
        model.reset()
        second = model.generate(prompt, max_new_tokens=16, temperature=0.0)
        if first.text != second.text:
            raise RuntimeError(
                f'greedy decoding was not deterministic: {first.text!r} vs {second.text!r}'
            )
        model.reset()
        constrained = model.generate(prompt, max_new_tokens=48, json_schema=SCHEMA)
        payload = json.loads(constrained.text)
        if 'colour' not in payload:
            raise RuntimeError(f'constrained reply missed a required key: {constrained.text!r}')
        print(
            f'generated {first.profile.generated_tokens} tokens '
            f'at {first.profile.decode_speed:.1f} tok/s; constrained output parsed'
        )
    finally:
        model.close()


if __name__ == '__main__':
    main()

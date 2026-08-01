#!/usr/bin/env python3
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Decode on the wheel's Vulkan backend, and prove it was really used.

Vulkan is the one GPU path that reaches NVIDIA, AMD, Intel and the mobile
vendors from a single build, so it is also the one most easily shipped broken:
the plugin loads either way, and a Vulkan build whose backend module was left
out of the wheel, or which cannot reach a driver, silently runs on the CPU
instead. Nothing about that failure is visible from the output text.

So this asserts the device, not just the answer. It refuses to pass if the
llama_cpp plugin reports no Vulkan device, and it pins the model to that
device by id rather than accepting whatever ``auto`` picks.

    python smoke_vulkan_wheel.py <model.gguf>

The counterpart is the plain smoke on a system with no Vulkan loader at all,
which proves the same wheel still runs on the CPU there -- the reason the
backends are separate loadable modules in the first place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: smoke_vulkan_wheel.py <model.gguf>')
    model_path = Path(sys.argv[1]).resolve()
    if not model_path.is_file():
        raise SystemExit(f'no such model file: {model_path}')

    # Exercise the installed package, not the checkout this script lives in.
    source_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != source_dir
    ]
    os.environ.pop('UNIRT_LIB_PATH', None)
    os.environ.pop('UNIRT_PLUGIN_PATH', None)

    from unirt import AutoModelForCausalLM, get_compute_unit_list, get_runtime_list, init

    init()
    runtimes = get_runtime_list()
    if 'llama_cpp' not in runtimes:
        # A Vulkan build that leaked a hard libvulkan dependency into the plugin
        # lands here rather than on the device check below, and the two want
        # very different fixes.
        raise SystemExit(
            f'the llama_cpp plugin did not load at all (runtimes: {runtimes}). '
            'A missing shared library is the usual cause; set the SDK log to '
            'debug for the loader error.'
        )
    devices = get_compute_unit_list('llama_cpp')
    for device_id, description in devices:
        print(f'  {device_id:<12} {description}')

    vulkan = [entry for entry in devices if entry[0].lower().startswith('vulkan')]
    if not vulkan:
        raise SystemExit(
            'the llama_cpp plugin reports no Vulkan device. Either the ggml Vulkan '
            'module is missing from the wheel, or it failed to load -- run with '
            'the SDK log at debug to see which.'
        )
    device_id = vulkan[0][0]

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map=f'llama_cpp:{device_id}', n_ctx=512
    )
    try:
        stats = model.runtime_stats()
        reported = stats.get('device_name') or ''
        # from_pretrained falls back to the CPU when an accelerator cannot be
        # initialised, which is right for users and useless for a test: without
        # this the job would pass on a machine where Vulkan never worked.
        if 'vulkan' not in reported.lower():
            raise SystemExit(
                f'asked for {device_id} but the model reports running on {reported!r}'
            )
        prompt = model._apply_chat_template(
            [{'role': 'user', 'content': 'Name one colour.'}], True, False, None
        )
        first = model.generate(prompt, max_new_tokens=16, temperature=0.0)
        if not first.text.strip():
            raise SystemExit('generation on the Vulkan device produced no text')
        model.reset()
        second = model.generate(prompt, max_new_tokens=16, temperature=0.0)
        if first.text != second.text:
            raise SystemExit(
                f'greedy decoding on Vulkan was not deterministic: '
                f'{first.text!r} vs {second.text!r}'
            )
        print(
            f'{reported}: generated {first.profile.generated_tokens} tokens at '
            f'{first.profile.decode_speed:.1f} tok/s'
        )
    finally:
        model.close()


if __name__ == '__main__':
    main()

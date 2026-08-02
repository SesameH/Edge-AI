# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive chat against a local model, on either UniRT backend.

Usage (from the repo root, after building — see README):

    python3 examples/chat.py --backend llama_cpp \
        --model models/SmolLM2-135M-Instruct-Q8_0.gguf
    python3 examples/chat.py --backend mlx \
        --model models/SmolLM2-135M-Instruct

The example automatically uses the local Python binding and locates either the
packaged or build-tree native runtime when executed from a source checkout.
"""

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_source_checkout() -> None:
    """Make the example runnable without installing the Python package."""
    repo_root = Path(__file__).resolve().parents[1]
    bindings = repo_root / 'bindings' / 'python'
    if bindings.is_dir() and str(bindings) not in sys.path:
        sys.path.insert(0, str(bindings))

    if 'UNIRT_LIB_PATH' in os.environ:
        return

    if sys.platform == 'darwin':
        library_name = 'libunirt.dylib'
    elif os.name == 'nt':
        library_name = 'unirt.dll'
    else:
        library_name = 'libunirt.so'

    packaged_library = repo_root / 'sdk' / 'pkg-unirt' / 'lib' / library_name
    build_library = repo_root / 'build' / 'src' / library_name
    if not packaged_library.is_file() and build_library.is_file():
        os.environ['UNIRT_LIB_PATH'] = str(build_library)
        plugin_dir = repo_root / 'build' / 'plugins'
        if plugin_dir.is_dir():
            os.environ.setdefault('UNIRT_PLUGIN_PATH', str(plugin_dir))


_bootstrap_source_checkout()

from unirt.auto import AutoModelForCausalLM  # noqa: E402


def _mb(n: int | None) -> str:
    return f'{n / 1024 / 1024:.0f}MB' if n is not None and n >= 0 else '?'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['llama_cpp', 'mlx'], default='llama_cpp')
    ap.add_argument(
        '--model',
        required=True,
        help='local model path or Hugging Face repository id',
    )
    ap.add_argument('--precision', help='remote GGUF quant, for example Q4_K_M')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--repeat-penalty', type=float, default=1.1)
    ap.add_argument('--seed', type=int, default=0, help='0 chooses a random seed')
    args = ap.parse_args()

    model_source = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    print(f'loading {model_source} on {args.backend} ...')
    with AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=args.backend,
        precision=args.precision,
    ) as model:
        stats = model.runtime_stats()
        print(f"device: {stats['device_name'] or 'unknown'}")
        print(
            f"memory: model {_mb(stats['model_bytes'])} · "
            f"process rss {_mb(stats['process_rss_bytes'])}"
        )
        print('ready. empty line or Ctrl-D to quit.\n')

        history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        while True:
            try:
                user = input('you> ').strip()
            except EOFError:
                break
            if not user:
                break
            history.append({'role': 'user', 'content': user})
            prompt = model._apply_chat_template(history, True, False, None)

            print('bot> ', end='', flush=True)
            streamer = model.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repeat_penalty,
                seed=args.seed,
                stream=True,
                # Keep talking when the conversation outgrows the context,
                # rather than stopping with an error. The backends will not
                # assume that on the caller's behalf.
                sliding_window=True,
            )
            reply = ''
            for chunk in streamer:
                reply += chunk
                print(chunk, end='', flush=True)
            print()
            out = streamer.output
            if out is not None:
                p = out.profile
                s = model.runtime_stats()
                mem = (
                    f"kv {_mb(s['kv_cache_bytes'])} · "
                    f"gpu peak {_mb(s['device_peak_bytes'])} · "
                    f"rss {_mb(s['process_rss_bytes'])}"
                )
                print(
                    f'     [{p.generated_tokens} tok · '
                    f'{p.decode_speed:.1f} tok/s · ttft {p.ttft/1000:.0f}ms · '
                    f'{p.stop_reason} | {mem}]'
                )
            history.append({'role': 'assistant', 'content': reply})
            # No reset between turns: both backends prefix-match the resent
            # transcript and only evaluate the new tokens, so each turn's
            # prefill cost stays proportional to the newest messages.


if __name__ == '__main__':
    sys.exit(main())

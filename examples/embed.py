# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Encode sentences and print their pairwise cosine similarity.

Usage (from the repo root, after building — see README):

    python3 examples/embed.py --model models/all-MiniLM-L6-v2-GGUF \
        "a cat sits on the mat" "a kitten rests on the rug" "quarterly revenue"

With no sentences, a small built-in demo set is encoded. Works with both
embedding backends: GGUF bundles run on llama_cpp, ONNX bundles on
onnxruntime — the format is detected from the model directory.
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

from unirt.auto import AutoModelForEmbedding  # noqa: E402

_DEMO_SENTENCES = [
    'a cat sits on the mat',
    'a kitten rests on the rug',
    'quarterly revenue rose sharply',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='embedding bundle directory (GGUF or ONNX)')
    ap.add_argument('--device', default='auto', help="device_map: auto / cpu / gpu / <plugin>:<dev>")
    ap.add_argument('sentences', nargs='*', help='sentences to encode (default: a demo set)')
    args = ap.parse_args()

    sentences = args.sentences or _DEMO_SENTENCES
    model = AutoModelForEmbedding.from_pretrained(args.model, device_map=args.device)
    try:
        vectors = model.encode(sentences)
        width = len(vectors[0]) if vectors else 0
        print(f'{len(vectors)} sentence(s), {width} dimensions each\n')
        if len(sentences) == 1:
            head = ', '.join(f'{value:+.4f}' for value in vectors[0][:8])
            print(f'  [{head}, ...]')
            return
        longest = max(len(s) for s in sentences)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = sum(x * y for x, y in zip(vectors[i], vectors[j]))
                print(f'  {sentences[i]:<{longest}}  ×  {sentences[j]:<{longest}}  cos = {similarity:+.4f}')
    finally:
        model.close()


if __name__ == '__main__':
    main()

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Encode text with the UniRT ONNX Runtime embedding backend."""

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bindings = repo_root / 'bindings' / 'python'
    if bindings.is_dir() and str(bindings) not in sys.path:
        sys.path.insert(0, str(bindings))
    if 'UNIRT_LIB_PATH' in os.environ:
        return
    library_name = (
        'libunirt.dylib' if sys.platform == 'darwin'
        else 'unirt.dll' if os.name == 'nt'
        else 'libunirt.so'
    )
    packaged = repo_root / 'sdk' / 'pkg-unirt' / 'lib' / library_name
    built = repo_root / 'build' / 'src' / library_name
    if not packaged.is_file() and built.is_file():
        os.environ['UNIRT_LIB_PATH'] = str(built)
        os.environ.setdefault('UNIRT_PLUGIN_PATH', str(repo_root / 'build' / 'plugins'))


_bootstrap_source_checkout()

from unirt import AutoModelForEmbedding  # noqa: E402


def _dot(left: list[float], right: list[float]) -> float:
    # AutoModelForEmbedding normalizes by default, so dot product is cosine.
    return sum(a * b for a, b in zip(left, right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='local ONNX bundle or Hugging Face repository id',
    )
    parser.add_argument('--device', choices=['cpu', 'coreml'], default='cpu')
    parser.add_argument('texts', nargs='+', help='one or more strings to encode')
    args = parser.parse_args()

    with AutoModelForEmbedding.from_pretrained(
        args.model,
        device_map=args.device,
    ) as model:
        vectors = model.encode(args.texts)
        assert isinstance(vectors, list) and vectors and isinstance(vectors[0], list)
        print(f'{len(vectors)} vectors × {len(vectors[0])} dimensions')
        for row, text in enumerate(args.texts):
            similarities = ' '.join(f'{_dot(vectors[row], vector):.4f}' for vector in vectors)
            print(f'[{row}] {similarities}  {text}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

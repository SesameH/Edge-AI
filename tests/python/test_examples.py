# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

from conftest import REPO_ROOT


def test_chat_example_imports_without_pythonpath():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)

    result = subprocess.run(
        [sys.executable, 'examples/chat.py', '--help'],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '--backend {llama_cpp,mlx}' in result.stdout
    assert '--precision PRECISION' in result.stdout
    assert '--repeat-penalty REPEAT_PENALTY' in result.stdout


def test_embedding_example_imports_without_pythonpath():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)

    result = subprocess.run(
        [sys.executable, 'examples/embedding.py', '--help'],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '--device {cpu,coreml}' in result.stdout
    assert '--model MODEL' in result.stdout

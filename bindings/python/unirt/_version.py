# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Package version kept dependency-free for build-time import."""

VERSION = (0, 1, 0)
__version__ = '.'.join(map(str, VERSION))

__all__ = ['__version__']

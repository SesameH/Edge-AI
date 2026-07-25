# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from unirt import catalog
from unirt import model_manager as mm

# The MLX plugin rejects anything that is not model_type=llama with tied
# embeddings and a [Digits, ByteLevel] BPE tokenizer, and it rejects it at load
# -- after the user has paid for the download. Recommending an unloadable
# checkpoint is therefore the one bug in this module that actually costs
# something, so the list of MLX-safe families is pinned here: widening it means
# widening the plugin's gates first.
_MLX_SAFE_PREFIXES = ('HuggingFaceTB/SmolLM2-',)


def test_aliases_are_unique_and_resolve_to_their_entry():
    aliases = catalog.aliases()
    assert len(aliases) == len(catalog.CATALOG)
    for entry in catalog.CATALOG:
        assert aliases[entry.alias] == entry.ref
        resolved, precision = mm._resolve_name(entry.alias, None)
        assert resolved == entry.repo
        assert precision == entry.quant


def test_catalog_aliases_do_not_shadow_repo_ids():
    for alias in catalog.aliases():
        assert '/' not in alias, f'{alias} would be parsed as a repo id, not an alias'


@pytest.mark.parametrize('entry', catalog.CATALOG, ids=lambda e: e.alias)
def test_entry_is_self_consistent(entry):
    assert entry.download_gib > 0
    assert entry.min_ram_gib > entry.download_gib
    assert entry.note.endswith('.')
    assert entry.backend in {'llama_cpp', 'mlx', 'onnxruntime'}
    # `repo:precision` is a GGUF-quant selector. MLX loads safetensors and ONNX
    # variants are picked at encode time, so a quant token on either backend
    # makes `pull` fail with "precision is not valid for safetensors repository".
    if entry.backend != 'llama_cpp':
        assert entry.quant is None


@pytest.mark.parametrize(
    'entry',
    [e for e in catalog.CATALOG if e.backend == 'mlx'],
    ids=lambda e: e.alias,
)
def test_mlx_entries_stay_inside_the_plugins_supported_families(entry):
    assert entry.repo.startswith(_MLX_SAFE_PREFIXES)
    # MLX loads HF safetensors directly; a GGUF quant token would never resolve.
    assert entry.quant is None


def test_recommend_filters_by_memory():
    tiny = catalog.recommend(ram_gib=2.5)
    assert tiny, 'a 2.5 GiB machine should still have something to run'
    assert all(entry.min_ram_gib <= 2.5 for entry in tiny)
    assert len(tiny) < len(catalog.CATALOG)


def test_recommend_include_oversized_ignores_memory():
    assert catalog.recommend(ram_gib=0.1) == []
    assert catalog.recommend(ram_gib=0.1, include_oversized=True) == list(
        catalog.recommend(ram_gib=None)
    )


def test_recommend_orders_best_fitting_model_first_within_a_task():
    chat = catalog.recommend(ram_gib=None, task='chat')
    sizes = [entry.download_gib for entry in chat]
    assert sizes == sorted(sizes, reverse=True)


def test_recommend_backend_filter():
    for backend in ('mlx', 'llama_cpp', 'onnxruntime'):
        selected = catalog.recommend(ram_gib=None, backend=backend)
        assert selected
        assert {entry.backend for entry in selected} == {backend}


def test_total_memory_is_plausible_or_unavailable():
    ram = catalog.total_memory_gib()
    assert ram is None or 0.25 < ram < 8192

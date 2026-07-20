# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-aware factories for local paths and Hugging Face repositories."""

from __future__ import annotations

import json
import logging
import os
import re
from ctypes import byref, c_void_p
from dataclasses import dataclass

from . import _progress
from . import model_manager as _mm
from ._ffi._api import (
    UniRTError,
    _check,
    ensure_init,
    get_compute_unit_list,
    get_runtime_list,
    load_library,
    resolve_device,
)
from ._ffi._types import (
    unirt_EmbeddingCreateInput,
    unirt_LlmCreateInput,
    unirt_ModelConfig,
    unirt_VlmCreateInput,
)
from .model_manager import ProgressCallback
from .modeling import UniRTEmbedding, UniRTLLM, UniRTVLM

PLUGIN_LLAMA_CPP = 'llama_cpp'
PLUGIN_MLX = 'mlx'
PLUGIN_ONNXRUNTIME = 'onnxruntime'

_RUNTIME_PRIORITY = (PLUGIN_LLAMA_CPP, PLUGIN_MLX)
_BUILTIN_VLM_PLUGINS = frozenset({PLUGIN_LLAMA_CPP})
_DEVICE_ALIASES = frozenset({'cpu', 'gpu', 'npu', 'hybrid'})
_ALIAS_OWNER = {alias: PLUGIN_LLAMA_CPP for alias in _DEVICE_ALIASES}

_SAFETENSOR_SHARD = re.compile(r'^model-\d+-of-\d+\.safetensors$', re.IGNORECASE)
_GGUF_SHARD = re.compile(r'-\d+-of-\d+\.gguf$', re.IGNORECASE)
_INT32_MAX = 2**31 - 1

# Compatibility with cache errors emitted by early native model managers.
UNIRT_ERROR_COMMON_UNKNOWN = -1000

_logger = logging.getLogger('unirt')


def _preferred_runtime(runtimes: list[str]) -> str | None:
    if not runtimes:
        return None
    return next(
        (runtime for runtime in _RUNTIME_PRIORITY if runtime in runtimes),
        sorted(runtimes)[0],
    )


def _apply_plugin_hint(device_map: str, plugin_id: str | None) -> str:
    if not isinstance(device_map, str) or '\x00' in device_map:
        raise ValueError('device_map must be a NUL-free string')
    requested = device_map.strip()
    if plugin_id is not None and (
        not isinstance(plugin_id, str) or '\x00' in plugin_id
    ):
        raise ValueError('plugin id must be a NUL-free string')
    if not plugin_id:
        return requested
    normalized = requested.lower()
    if not requested or normalized == 'auto':
        return plugin_id
    if normalized in _DEVICE_ALIASES:
        return f'{plugin_id}:{normalized}'
    return requested


def _call_sdk(
    plugin_id: str,
    alias: str | None,
) -> tuple[str, str | None, int | None]:
    device_id, n_gpu_layers, warning = resolve_device(plugin_id, alias, -1)
    if warning:
        _logger.warning('%s', warning)
    override = None if n_gpu_layers == -1 else n_gpu_layers
    return plugin_id, device_id, override


def resolve_device_map(
    device_map: str,
) -> tuple[str | None, str | None, int | None]:
    """Resolve auto, aliases, plugin ids, and ``plugin:device`` strings."""
    if not isinstance(device_map, str) or '\x00' in device_map:
        raise ValueError('device_map must be a NUL-free string')
    requested = device_map.strip()
    normalized = requested.lower()

    if not requested or normalized == 'auto':
        runtime = _preferred_runtime(get_runtime_list())
        return (None, None, None) if runtime is None else _call_sdk(runtime, None)

    if normalized in _DEVICE_ALIASES:
        runtimes = get_runtime_list()
        owner = _ALIAS_OWNER[normalized]
        runtime = owner if owner in runtimes else _preferred_runtime(runtimes)
        if runtime is None:
            runtime = PLUGIN_LLAMA_CPP
        return _call_sdk(runtime, normalized)

    if ':' not in requested:
        return _call_sdk(requested, None)

    plugin_id, device_id = (part.strip() for part in requested.split(':', 1))
    if not plugin_id or not device_id:
        raise ValueError(
            "device_map must use '<runtime>:<compute-unit>' with both parts non-empty"
        )
    if device_id.lower() in _DEVICE_ALIASES:
        return _call_sdk(plugin_id, device_id.lower())
    return plugin_id, device_id, None


def _is_mmproj_name(name: str) -> bool:
    stem = os.path.basename(name).lower()
    if stem.endswith('.gguf'):
        stem = stem[:-5]
    return (
        stem == 'mmproj'
        or stem.startswith(('mmproj-', 'mmproj.', 'mmproj_'))
        or stem.endswith(('-mmproj', '_mmproj'))
        or '-mmproj-' in stem
        or '_mmproj_' in stem
    )


def _files_in(directory: str) -> list[str]:
    try:
        return sorted(
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        )
    except OSError:
        return []


def _resolve_local_anchor(path: str) -> str:
    if not os.path.isdir(path):
        return path
    entries = _files_in(path)
    if not entries:
        raise FileNotFoundError(f'No files found in model directory: {path}')
    if 'model.safetensors' in entries:
        return os.path.join(path, 'model.safetensors')

    safetensor_shards = [name for name in entries if _SAFETENSOR_SHARD.match(name)]
    if safetensor_shards:
        return os.path.join(path, safetensor_shards[0])

    ggufs = [
        name for name in entries
        if name.lower().endswith('.gguf') and not _is_mmproj_name(name)
    ]
    one_shard_family = (
        bool(ggufs)
        and all(_GGUF_SHARD.search(name) for name in ggufs)
        and len({_GGUF_SHARD.sub('.gguf', name) for name in ggufs}) == 1
    )
    if len(ggufs) == 1 or one_shard_family:
        return os.path.join(path, ggufs[0])
    if ggufs:
        raise ValueError(
            'model directory contains multiple GGUF variants; pass the desired '
            f'GGUF file explicitly: {path}'
        )

    if 'tokenizer.json' in entries:
        return os.path.join(path, 'tokenizer.json')
    return os.path.join(path, entries[0])


def _local_sidecars(path: str) -> tuple[str | None, str | None]:
    directory = path if os.path.isdir(path) else (os.path.dirname(path) or os.curdir)
    entries = _files_in(directory)
    projectors = [
        os.path.join(directory, name) for name in entries if _is_mmproj_name(name)
    ]
    q8_projectors = [
        projector for projector in projectors
        if _mm._extract_quant_token(os.path.basename(projector)) == 'Q8_0'
    ]
    candidates = q8_projectors or projectors
    mmproj = min(
        candidates,
        key=lambda item: (os.path.getsize(item), item),
        default=None,
    )
    tokenizer = os.path.join(directory, 'tokenizer.json')
    return mmproj, tokenizer if os.path.isfile(tokenizer) else None


def _runtime_for_model_path(model_path: str) -> str | None:
    absolute = os.path.abspath(model_path)
    if os.path.isfile(absolute) and absolute.lower().endswith('.gguf'):
        return PLUGIN_LLAMA_CPP
    directory = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
    names = _files_in(directory)
    if any(
        name.lower() == 'model.safetensors' or _SAFETENSOR_SHARD.match(name)
        for name in names
    ):
        return PLUGIN_MLX
    if any(
        name.lower().endswith('.gguf') and not _is_mmproj_name(name)
        for name in names
    ):
        return PLUGIN_LLAMA_CPP
    return None


def _read_json_object(path: str) -> dict | None:
    try:
        with open(path, encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _bundle_directory(model_path: str) -> str:
    return (
        model_path
        if os.path.isdir(model_path)
        else (os.path.dirname(model_path) or os.curdir)
    )


def _read_bundle_metadata(model_path: str) -> dict | None:
    return _read_json_object(os.path.join(_bundle_directory(model_path), 'metadata.json'))


def _detect_supports_thinking(
    model_path: str | None,
    tokenizer_path: str | None,
) -> bool | None:
    candidates: list[str] = []
    if tokenizer_path:
        candidates.append(
            os.path.join(os.path.dirname(tokenizer_path), 'tokenizer_config.json')
        )
    if model_path:
        candidates.append(
            os.path.join(_bundle_directory(model_path), 'tokenizer_config.json')
        )
    for candidate in dict.fromkeys(candidates):
        config = _read_json_object(candidate)
        if config is None:
            continue
        template = config.get('chat_template')
        if not isinstance(template, str):
            return None
        return 'enable_thinking' in template or '<think>' in template
    return None


def _is_vlm(
    mmproj_path: str | None,
    cache_key: str,
    model_path: str | None = None,
) -> bool:
    if mmproj_path is not None:
        return True
    if model_path:
        directory = _bundle_directory(model_path)
        config = _read_json_object(os.path.join(directory, 'config.json'))
        if config is not None and _mm._model_type(_files_in(directory), config) == 'vlm':
            return True
    try:
        return _mm.get_type(cache_key) == 'vlm'
    except Exception:  # an uncached or malformed bundle is not enough evidence
        return False


def _validate_runtime_for_model(plugin_id: str | None, model_path: str) -> None:
    inferred = _runtime_for_model_path(model_path)
    if (
        inferred
        and plugin_id in {PLUGIN_LLAMA_CPP, PLUGIN_MLX}
        and plugin_id != inferred
    ):
        model_format = 'GGUF' if inferred == PLUGIN_LLAMA_CPP else 'safetensors'
        raise ValueError(
            f"backend {plugin_id!r} cannot load this {model_format} model; "
            f"use backend {inferred!r}"
        )


def _require_vlm_backend(plugin_id: str | None) -> None:
    if (
        plugin_id in {PLUGIN_LLAMA_CPP, PLUGIN_MLX}
        and plugin_id not in _BUILTIN_VLM_PLUGINS
    ):
        raise NotImplementedError(
            f"backend {plugin_id!r} is text-only in this build; no bundled UniRT "
            'plugin implements IVlm. Select llama_cpp or install a VLM-capable plugin.'
        )


def _require_available_backend(plugin_id: str | None) -> None:
    if not plugin_id:
        raise RuntimeError('no UniRT inference backend is installed or available')
    if plugin_id == PLUGIN_MLX and not get_compute_unit_list(plugin_id):
        raise RuntimeError(
            'the MLX backend is installed but no usable Apple Metal device is '
            'available; use llama_cpp on CPU or run MLX in a Metal-capable macOS process'
        )


def _validate_optional_text(label: str, value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or not value or '\x00' in value
    ):
        raise ValueError(f'{label} must be a non-empty NUL-free string or None')


def _resolve_model_sources(
    model_name_or_path: str,
    quant: str | None,
    hf_token: str | None,
    progress: ProgressCallback | bool | None,
    model_name: str | None = None,
) -> tuple[str, str | None, str | None, _mm.ModelFiles | None]:
    if (
        not isinstance(model_name_or_path, str)
        or not model_name_or_path
        or '\x00' in model_name_or_path
    ):
        raise ValueError('model_name_or_path must be a non-empty NUL-free string')
    _validate_optional_text('quant', quant)
    _validate_optional_text('model_name', model_name)

    if os.path.exists(model_name_or_path):
        if model_name:
            metadata = _read_bundle_metadata(model_name_or_path) or {}
            bundle_id = metadata.get('model_id')
            if bundle_id and bundle_id != model_name:
                raise ValueError(
                    f"model_name '{model_name}' does not match bundle "
                    f"'model_id={bundle_id}' in {model_name_or_path}"
                )
            local_dir = _bundle_directory(model_name_or_path)
            local_precision = quant
            if local_precision is None and os.path.isfile(model_name_or_path):
                local_precision = _mm._extract_quant(os.path.basename(model_name_or_path))
            _mm.pull(
                model_name,
                precision=local_precision,
                hub='localfs',
                local_path=os.path.abspath(local_dir),
            )
            paths = _mm.get_paths(model_name)
            return paths.model_path, paths.mmproj_path, paths.tokenizer_path, paths

        mmproj, tokenizer = _local_sidecars(model_name_or_path)
        return _resolve_local_anchor(model_name_or_path), mmproj, tokenizer, None

    cache_key = f'{model_name_or_path}:{quant}' if quant else model_name_or_path
    try:
        cached = _mm.get_paths(cache_key)
        return cached.model_path, cached.mmproj_path, cached.tokenizer_path, cached
    except (UniRTError, FileNotFoundError, OSError):
        pass

    printer = _progress.resolve(progress)
    try:
        try:
            paths = _mm.ensure_cached(
                model_name_or_path,
                precision=quant,
                hub='auto',
                hf_token=hf_token,
                on_progress=printer,
            )
        except UniRTError as error:
            translated = _translate_quant_error(error, model_name_or_path, quant)
            if translated is not None:
                raise translated from error
            raise
    finally:
        _progress.finish(printer)
    return paths.model_path, paths.mmproj_path, paths.tokenizer_path, paths


def _translate_quant_error(
    error: UniRTError,
    model_name_or_path: str,
    quant: str | None,
) -> ValueError | None:
    if quant is None or error.code != UNIRT_ERROR_COMMON_UNKNOWN:
        return None
    return ValueError(f'Could not resolve quant {quant!r} for {model_name_or_path!r}.')


_MODEL_INT_FIELDS = frozenset({
    'n_threads',
    'n_threads_batch',
    'n_batch',
    'n_ubatch',
    'n_seq_max',
})
_MODEL_TEXT_FIELDS = frozenset({
    'chat_template_path',
    'chat_template_content',
    'grammar_str',
})


def _checked_int(name: str, value, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    if not minimum <= value <= _INT32_MAX:
        raise ValueError(f'{name} must be between {minimum} and {_INT32_MAX}')
    return value


def _build_model_config(
    n_ctx: int,
    n_gpu_layers: int,
    **kwargs,
) -> unirt_ModelConfig:
    config = unirt_ModelConfig(
        n_ctx=_checked_int('n_ctx', n_ctx),
        n_gpu_layers=_checked_int('n_gpu_layers', n_gpu_layers, -1),
    )
    for name, value in kwargs.items():
        if name in _MODEL_INT_FIELDS:
            setattr(config, name, _checked_int(name, value))
        elif name in _MODEL_TEXT_FIELDS:
            if value is not None and (
                not isinstance(value, str) or '\x00' in value
            ):
                raise ValueError(f'{name} must be a NUL-free string or None')
            if value is not None:
                setattr(config, name, value.encode('utf-8'))
        else:
            raise TypeError(f'unknown model configuration argument: {name}')
    return config


@dataclass(frozen=True)
class _LoadPlan:
    resolved_name: str
    model_path: str
    mmproj_path: str | None
    tokenizer_path: str | None
    plugin_id: str | None
    device_id: str | None
    config: unirt_ModelConfig
    meta: dict


def _prepare_load(
    model_name_or_path: str,
    model_name: str | None,
    precision: str | None,
    device_map: str,
    n_ctx: int,
    n_gpu_layers: int,
    mmproj_path: str | None,
    tokenizer_path: str | None,
    hf_token: str | None,
    progress: ProgressCallback | bool | None,
    kwargs: dict,
) -> _LoadPlan:
    ensure_init()
    model_path, discovered_mmproj, discovered_tokenizer, paths = _resolve_model_sources(
        model_name_or_path,
        precision,
        hf_token,
        progress,
        model_name,
    )
    resolved_name = (
        model_name
        or (paths.model_name if paths and paths.model_name else None)
        or model_name_or_path
    )
    runtime_hint = (paths.runtime if paths else None) or _runtime_for_model_path(model_path)
    requested_device = _apply_plugin_hint(device_map, runtime_hint)
    plugin_id, device_id, layer_override = resolve_device_map(requested_device)
    _validate_runtime_for_model(plugin_id, model_path)
    _require_available_backend(plugin_id)
    if layer_override is not None:
        n_gpu_layers = layer_override

    resolved_tokenizer = tokenizer_path or discovered_tokenizer
    config = _build_model_config(n_ctx, n_gpu_layers, **kwargs)
    return _LoadPlan(
        resolved_name=resolved_name,
        model_path=model_path,
        mmproj_path=mmproj_path or discovered_mmproj,
        tokenizer_path=resolved_tokenizer,
        plugin_id=plugin_id,
        device_id=device_id,
        config=config,
        meta={
            'model_name': resolved_name,
            'backend': plugin_id,
            'device': device_id,
            'quant': precision,
            'model_path': model_path,
            'supports_thinking': _detect_supports_thinking(
                model_path,
                resolved_tokenizer,
            ),
        },
    )


def _create_llm_handle(plan: _LoadPlan) -> UniRTLLM:
    input_value = unirt_LlmCreateInput(
        model_path=plan.model_path.encode('utf-8'),
        tokenizer_path=(
            plan.tokenizer_path.encode('utf-8') if plan.tokenizer_path else None
        ),
        config=plan.config,
        plugin_id=plan.plugin_id.encode('utf-8') if plan.plugin_id else None,
        device_id=plan.device_id.encode('utf-8') if plan.device_id else None,
    )
    handle = c_void_p()
    _check(load_library().unirt_llm_create(byref(input_value), byref(handle)))
    return UniRTLLM(handle, meta=plan.meta)


def _create_vlm_handle(
    resolved_name: str,
    model_path: str,
    mmproj_path: str | None,
    tokenizer_path: str | None,
    plugin_id: str | None,
    device_id: str | None,
    config: unirt_ModelConfig,
    meta: dict | None = None,
) -> UniRTVLM:
    input_value = unirt_VlmCreateInput(
        model_path=model_path.encode('utf-8'),
        mmproj_path=mmproj_path.encode('utf-8') if mmproj_path else None,
        config=config,
        plugin_id=plugin_id.encode('utf-8') if plugin_id else None,
        device_id=device_id.encode('utf-8') if device_id else None,
        tokenizer_path=tokenizer_path.encode('utf-8') if tokenizer_path else None,
    )
    handle = c_void_p()
    _check(load_library().unirt_vlm_create(byref(input_value), byref(handle)))
    return UniRTVLM(handle, meta=meta)


def _create_vlm_from_plan(plan: _LoadPlan) -> UniRTVLM:
    return _create_vlm_handle(
        plan.resolved_name,
        plan.model_path,
        plan.mmproj_path,
        plan.tokenizer_path,
        plan.plugin_id,
        plan.device_id,
        plan.config,
        meta=plan.meta,
    )


_EMBEDDING_POOLING = {
    'default': 0,
    'model_default': 0,
    'cls': 1,
    'mean': 2,
    'last_token': 3,
}


def _embedding_precision(value: str | None) -> str:
    if value is None:
        return 'onnx'
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('precision must be a non-empty NUL-free string or None')
    normalized = re.sub(r'[^a-z0-9]+', '-', value.casefold()).strip('-')
    if normalized in {'default', 'fp32', 'onnx', 'onnx-fp32'}:
        return 'onnx'
    if normalized == 'gguf' or normalized.startswith('gguf-'):
        return normalized
    return normalized if normalized.startswith('onnx-') else f'onnx-{normalized}'


def _find_embedding_model(directory: str, precision: str) -> str:
    onnx_candidates: list[str] = []
    gguf_candidates: list[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in {'.cache', '.git'}]
        for name in files:
            lowered = name.casefold()
            if lowered.endswith('.onnx'):
                onnx_candidates.append(os.path.join(root, name))
            elif lowered.endswith('.gguf'):
                gguf_candidates.append(os.path.join(root, name))
    if not onnx_candidates and not gguf_candidates:
        raise FileNotFoundError(f'No ONNX or GGUF model found in directory: {directory}')

    def variant(path: str) -> str:
        if path.casefold().endswith('.gguf'):
            return 'gguf'
        stem = os.path.splitext(os.path.basename(path))[0].casefold()
        if stem in {'model', 'model-fp32', 'model_fp32'}:
            return 'onnx'
        for prefix in ('model-', 'model_'):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        suffix = re.sub(r'[^a-z0-9]+', '-', stem).strip('-')
        return f'onnx-{suffix}'

    candidates = onnx_candidates + gguf_candidates
    matching = [path for path in candidates if variant(path) == precision]
    if not matching and precision == 'onnx' and not onnx_candidates:
        # The default precision on a GGUF-only bundle means "use the GGUF".
        matching = gguf_candidates
    if not matching:
        available = ', '.join(sorted({variant(path) for path in candidates}))
        raise ValueError(
            f'embedding precision {precision!r} not found in {directory}; available: {available}'
        )
    return min(
        matching,
        key=lambda path: (
            os.path.relpath(path, directory).count(os.sep),
            len(os.path.relpath(path, directory)),
            path,
        ),
    )


def _find_tokenizer(directory: str) -> str | None:
    direct = os.path.join(directory, 'tokenizer.json')
    if os.path.isfile(direct):
        return direct
    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in {'.cache', '.git'}]
        if 'tokenizer.json' in files:
            matches.append(os.path.join(root, 'tokenizer.json'))
    return min(matches, key=lambda path: (path.count(os.sep), path), default=None)


def _resolve_embedding_sources(
    model_name_or_path: str,
    precision: str | None,
    tokenizer_path: str | None,
    hf_token: str | None,
    progress: ProgressCallback | bool | None,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(model_name_or_path, str)
        or not model_name_or_path
        or '\x00' in model_name_or_path
    ):
        raise ValueError('model_name_or_path must be a non-empty NUL-free string')
    wanted = _embedding_precision(precision)
    if os.path.exists(model_name_or_path):
        absolute = os.path.abspath(model_name_or_path)
        if os.path.isdir(absolute):
            model_path = _find_embedding_model(absolute, wanted)
            bundle_dir = absolute
        elif absolute.casefold().endswith('.onnx'):
            model_path = absolute
            bundle_dir = os.path.dirname(absolute)
        else:
            raise ValueError('embedding model must be an .onnx file or a directory')
        resolved_tokenizer = tokenizer_path or _find_tokenizer(bundle_dir)
        if resolved_tokenizer is None and os.path.basename(bundle_dir).casefold() == 'onnx':
            resolved_tokenizer = _find_tokenizer(os.path.dirname(bundle_dir))
        if not resolved_tokenizer or not os.path.isfile(resolved_tokenizer):
            raise FileNotFoundError('embedding bundle must contain tokenizer.json')
        return model_path, os.path.abspath(resolved_tokenizer), model_name_or_path, bundle_dir

    printer = _progress.resolve(progress)
    try:
        paths = _mm.ensure_cached(
            model_name_or_path,
            precision=wanted,
            hub='auto',
            hf_token=hf_token,
            on_progress=printer,
        )
    finally:
        _progress.finish(printer)
    resolved_tokenizer = tokenizer_path or paths.tokenizer_path
    if not resolved_tokenizer or not os.path.isfile(resolved_tokenizer):
        raise FileNotFoundError(
            f'Hugging Face embedding bundle {model_name_or_path!r} has no tokenizer.json'
        )
    return paths.model_path, resolved_tokenizer, paths.model_name, paths.model_dir


def _embedding_max_length(bundle_dir: str, tokenizer_path: str, requested: int | None) -> int:
    if requested is not None:
        return _checked_int('max_length', requested, 1)
    directories = [os.path.dirname(tokenizer_path), bundle_dir]
    values: list[int] = []
    for directory in dict.fromkeys(directories):
        sentence_config = _read_json_object(os.path.join(directory, 'sentence_bert_config.json'))
        tokenizer_config = _read_json_object(os.path.join(directory, 'tokenizer_config.json'))
        model_config = _read_json_object(os.path.join(directory, 'config.json'))
        for config, key in (
            (sentence_config, 'max_seq_length'),
            (tokenizer_config, 'model_max_length'),
            (model_config, 'max_position_embeddings'),
        ):
            value = config.get(key) if config else None
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= _INT32_MAX:
                values.append(value)
    return min(values) if values else 512


def _resolve_embedding_device(device_map: str, model_name: str, model_path: str) -> tuple[str, str]:
    if not isinstance(device_map, str) or not device_map or '\x00' in device_map:
        raise ValueError('device_map must be a non-empty NUL-free string')
    if model_path.casefold().endswith('.gguf'):
        return _resolve_gguf_embedding_device(device_map, model_name)
    if PLUGIN_ONNXRUNTIME not in get_runtime_list():
        raise RuntimeError('the ONNX Runtime plugin is not installed or failed to load')
    requested = device_map.strip().casefold()
    if requested in {'auto', PLUGIN_ONNXRUNTIME, 'cpu', f'{PLUGIN_ONNXRUNTIME}:cpu'}:
        device = 'cpu'
    elif requested in {'gpu', 'coreml', f'{PLUGIN_ONNXRUNTIME}:gpu', f'{PLUGIN_ONNXRUNTIME}:coreml'}:
        device = 'coreml'
    elif ':' in requested:
        plugin, device = requested.split(':', 1)
        if plugin != PLUGIN_ONNXRUNTIME or device not in {'cpu', 'coreml'}:
            raise ValueError("embedding device_map must be 'cpu', 'coreml', or 'onnxruntime:<device>'")
    else:
        raise ValueError("embedding device_map must be 'cpu', 'coreml', or 'auto'")
    available = {identifier for identifier, _ in get_compute_unit_list(PLUGIN_ONNXRUNTIME)}
    if device not in available:
        raise RuntimeError(f'ONNX Runtime device {device!r} is unavailable for {model_name!r}')
    return PLUGIN_ONNXRUNTIME, device


def _resolve_gguf_embedding_device(device_map: str, model_name: str) -> tuple[str, str]:
    if PLUGIN_LLAMA_CPP not in get_runtime_list():
        raise RuntimeError(
            f'{model_name!r} is a GGUF embedding model but the llama_cpp plugin is not available'
        )
    requested = device_map.strip().casefold()
    if requested in {'auto', 'gpu', PLUGIN_LLAMA_CPP, f'{PLUGIN_LLAMA_CPP}:gpu'}:
        # Empty device id lets llama.cpp pick its default (Metal on Apple).
        return PLUGIN_LLAMA_CPP, ''
    if requested in {'cpu', f'{PLUGIN_LLAMA_CPP}:cpu'}:
        return PLUGIN_LLAMA_CPP, 'cpu'
    if ':' in requested:
        plugin, device = requested.split(':', 1)
        if plugin == PLUGIN_LLAMA_CPP and device:
            return PLUGIN_LLAMA_CPP, device
    raise ValueError(
        "GGUF embedding device_map must be 'auto', 'cpu', 'gpu', or 'llama_cpp:<device>'"
    )


class AutoModelForEmbedding:
    """Load a text encoder from a local or Hugging Face ONNX bundle."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_name: str | None = None,
        precision: str | None = None,
        device_map: str = 'auto',
        tokenizer_path: str | None = None,
        max_length: int | None = None,
        pooling: str = 'mean',
        normalize: bool = True,
        output_name: str | None = None,
        hf_token: str | None = None,
        progress: ProgressCallback | bool | None = None,
    ) -> UniRTEmbedding:
        ensure_init()
        _validate_optional_text('model_name', model_name)
        _validate_optional_text('tokenizer_path', tokenizer_path)
        _validate_optional_text('output_name', output_name)
        if not isinstance(pooling, str) or pooling.casefold() not in _EMBEDDING_POOLING:
            raise ValueError(f"pooling must be one of: {', '.join(sorted(_EMBEDDING_POOLING))}")
        if not isinstance(normalize, bool):
            raise TypeError('normalize must be a boolean')

        model_path, resolved_tokenizer, discovered_name, bundle_dir = _resolve_embedding_sources(
            model_name_or_path,
            precision,
            tokenizer_path,
            hf_token,
            progress,
        )
        resolved_name = model_name or discovered_name
        plugin_id, device_id = _resolve_embedding_device(device_map, resolved_name, model_path)
        resolved_max_length = _embedding_max_length(bundle_dir, resolved_tokenizer, max_length)
        input_value = unirt_EmbeddingCreateInput(
            model_path=model_path.encode('utf-8'),
            plugin_id=plugin_id.encode('utf-8'),
            device_id=device_id.encode('utf-8'),
            pooling=_EMBEDDING_POOLING[pooling.casefold()],
            normalize=normalize,
            output_name=output_name.encode('utf-8') if output_name else None,
        )
        handle = c_void_p()
        _check(load_library().unirt_embedding_create(byref(input_value), byref(handle)))
        return UniRTEmbedding(
            handle,
            resolved_tokenizer,
            max_length=resolved_max_length,
            meta={
                'model_name': resolved_name,
                'backend': plugin_id,
                'device': device_id,
                'precision': 'gguf'
                if model_path.casefold().endswith('.gguf')
                else _embedding_precision(precision),
                'model_path': model_path,
            },
        )


class AutoModelForCausalLM:
    """Load a language model, auto-promoting GGUF+mmproj bundles to VLM."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_name: str | None = None,
        precision: str | None = None,
        device_map: str = 'auto',
        n_ctx: int = 0,
        n_gpu_layers: int = -1,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        hf_token: str | None = None,
        progress: ProgressCallback | bool | None = None,
        **kwargs,
    ) -> UniRTLLM | UniRTVLM:
        plan = _prepare_load(
            model_name_or_path,
            model_name,
            precision,
            device_map,
            n_ctx,
            n_gpu_layers,
            mmproj_path,
            tokenizer_path,
            hf_token,
            progress,
            kwargs,
        )
        cache_key = model_name or model_name_or_path
        if _is_vlm(plan.mmproj_path, cache_key, plan.model_path):
            _require_vlm_backend(plan.plugin_id)
            return _create_vlm_from_plan(plan)
        return _create_llm_handle(plan)


class AutoModelForVision2Seq:
    """Load a vision-language or other multimodal model."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_name: str | None = None,
        precision: str | None = None,
        device_map: str = 'auto',
        n_ctx: int = 0,
        n_gpu_layers: int = -1,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        hf_token: str | None = None,
        progress: ProgressCallback | bool | None = None,
        **kwargs,
    ) -> UniRTVLM:
        plan = _prepare_load(
            model_name_or_path,
            model_name,
            precision,
            device_map,
            n_ctx,
            n_gpu_layers,
            mmproj_path,
            tokenizer_path,
            hf_token,
            progress,
            kwargs,
        )
        _require_vlm_backend(plan.plugin_id)
        return _create_vlm_from_plan(plan)


def _infer_model_kind(model_name_or_path: str) -> str:
    """Best-effort local sniff: embedding bundles carry an ONNX graph or a
    sentence-transformers config; everything else loads as a language model
    (which auto-upgrades to a VLM when a projector is present)."""
    absolute = os.path.abspath(model_name_or_path)
    directory = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
    names = {name.lower() for name in _files_in(directory)}
    if any(name.endswith('.onnx') for name in names) or 'sentence_bert_config.json' in names:
        return 'embedding'
    return 'llm'


def load(
    model_name_or_path: str,
    *,
    kind: str = 'auto',
    **kwargs,
):
    """UniRT's native entry point: load any supported model in one call.

    ``kind`` selects the modality — ``"llm"``, ``"vlm"``, ``"embedding"``,
    or ``"auto"`` (default) to detect it from the model files. Keyword
    arguments are forwarded to the matching ``from_pretrained``. The
    HF-style ``AutoModelFor*`` classes remain available as aliases for
    callers porting code from transformers.
    """
    if not isinstance(kind, str):
        raise TypeError('kind must be a string')
    normalized = kind.strip().casefold()
    if normalized not in {'auto', 'llm', 'vlm', 'embedding'}:
        raise ValueError("kind must be one of: auto, llm, vlm, embedding")
    if normalized == 'auto':
        normalized = _infer_model_kind(model_name_or_path)
    if normalized == 'embedding':
        return AutoModelForEmbedding.from_pretrained(model_name_or_path, **kwargs)
    if normalized == 'vlm':
        return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


__all__ = [
    'AutoModelForCausalLM',
    'AutoModelForEmbedding',
    'AutoModelForVision2Seq',
    'load',
    'resolve_device_map',
]

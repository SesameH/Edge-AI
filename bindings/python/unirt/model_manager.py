# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Small, Python-native model store backed by :mod:`huggingface_hub`.

The module deliberately keeps the public API of the former native model
manager.  Downloading and Hub authentication are delegated to
``huggingface_hub``; UniRT only decides which model files belong together and
writes a tiny ``unirt.json`` index next to them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, tqdm as hub_tqdm

from ._ffi._api import UniRTError

__all__ = [
    'ModelFiles',
    'CachedModel',
    'PrecisionVariant',
    'ModelInspection',
    'DownloadProgress',
    'ProgressCallback',
    'init',
    'deinit',
    'pull',
    'list_models',
    'list_detailed',
    'last_error_message',
    'query',
    'remove',
    'clean',
    'get_paths',
    'get_type',
    'set_type',
    'ensure_cached',
]


# The only model sources UniRT supports.
UNIRT_HUB_AUTO = 0
UNIRT_HUB_HUGGINGFACE = 1
UNIRT_HUB_LOCALFS = 127

UNIRT_ERROR_COMMON_UNKNOWN = -1000
UNIRT_ERROR_COMMON_INVALID_INPUT = -1001
UNIRT_ERROR_COMMON_FILE_NOT_FOUND = -1004
UNIRT_ERROR_COMMON_NETWORK = -1100
UNIRT_ERROR_COMMON_CANCELLED = -1007
UNIRT_ERROR_COMMON_NOT_INITIALIZED = -1005
UNIRT_ERROR_COMMON_AUTH = -1101
UNIRT_ERROR_COMMON_HUB_MODEL_NOT_FOUND = -1102
UNIRT_ERROR_COMMON_NOT_SUPPORTED = -1008
UNIRT_ERROR_COMMON_MANIFEST_PARSE = -1105

_MANIFEST = 'unirt.json'
_QUANT_PRIORITY = ('Q4_0', 'Q4_K_M', 'Q8_0')
_ALIASES = {'qwen3': 'ggml-org/Qwen3-1.7B-GGUF:Q4_K_M'}
_SIDECAR_SUFFIXES = (
    '.json',
    '.jinja',
    '.model',
    '.tiktoken',
    '.txt',
)
_VLM_ARCH_TOKENS = (
    'aria',
    'fuyu',
    'idefics',
    'internvl',
    'llama4',
    'llava',
    'minicpmv',
    'mllama',
    'paligemma',
    'vision',
    'vl',
)
_QUANT_RE = re.compile(
    r'(?<![A-Z0-9])('
    r'(?:I|T)?Q\d+(?:_[A-Z0-9]+){0,3}'
    r'|MXFP\d+(?:_[A-Z0-9]+){0,2}'
    r')(?![A-Z0-9])',
    re.IGNORECASE,
)
_HF_SAFETENSOR_SHARD_RE = re.compile(r'^model-(\d+)-of-(\d+)\.safetensors$', re.IGNORECASE)
_NUMBERED_SHARD_RE = re.compile(r'-(\d+)-of-(\d+)(?=\.[^.]+$)', re.IGNORECASE)
_REPO_COMPONENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')


@dataclass(frozen=True)
class DownloadProgress:
    """Per-file transfer progress."""

    file_name: str
    downloaded_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class ModelFiles:
    """Resolved absolute paths for one cached model variant."""

    model_path: str
    model_dir: str
    model_name: str
    runtime: str
    model_type: str
    mmproj_path: str | None = None
    tokenizer_path: str | None = None


@dataclass(frozen=True)
class CachedModel:
    """Summary of a locally cached repository."""

    name: str
    model_name: str
    runtime: str
    model_type: str
    total_size: int
    precisions: list[str]


@dataclass(frozen=True)
class PrecisionVariant:
    """A model variant advertised by a Hugging Face repository."""

    precision: str
    size: int
    is_default: bool


@dataclass(frozen=True)
class ModelInspection:
    """Result of inspecting a repository without downloading it."""

    model_name: str
    runtime: str
    model_type: str
    candidates: list[PrecisionVariant]


ProgressCallback = Callable[[list[DownloadProgress]], bool]


@dataclass(frozen=True)
class _RemoteFile:
    name: str
    size: int


@dataclass(frozen=True)
class _Plan:
    repo_id: str
    model_name: str
    runtime: str
    model_type: str
    precision: str
    model_files: tuple[_RemoteFile, ...]
    shared_files: tuple[_RemoteFile, ...]
    mmproj: _RemoteFile | None
    tokenizer: _RemoteFile | None
    candidates: tuple[PrecisionVariant, ...]

    @property
    def files(self) -> tuple[_RemoteFile, ...]:
        unique: dict[str, _RemoteFile] = {}
        for item in (*self.model_files, *self.shared_files):
            unique[item.name] = item
        if self.mmproj:
            unique[self.mmproj.name] = self.mmproj
        if self.tokenizer:
            unique[self.tokenizer.name] = self.tokenizer
        return tuple(unique.values())


_state_lock = threading.RLock()
_progress_lock = threading.RLock()
_thread_state = threading.local()
_data_dir: Path | None = None


def _set_error(message: str | None) -> None:
    _thread_state.last_error = message


def _raise(code: int, message: str):
    _set_error(message)
    raise UniRTError(code, message)


def last_error_message() -> str | None:
    """Return the most recent model-manager error on this thread."""

    return getattr(_thread_state, 'last_error', None)


def init(data_dir: str | None = None) -> None:
    """Initialize the local store. This operation is idempotent."""

    global _data_dir
    with _state_lock:
        if _data_dir is not None:
            return
        root = data_dir or os.environ.get('UNIRT_DATADIR')
        _data_dir = Path(root).expanduser() if root else Path.home() / '.cache' / 'unirt'
        (_data_dir / 'models').mkdir(parents=True, exist_ok=True)


def deinit() -> None:
    """Forget process-local store state; downloaded files are untouched."""

    global _data_dir
    with _state_lock:
        _data_dir = None


def _root() -> Path:
    if _data_dir is None:
        init()
    assert _data_dir is not None
    return _data_dir


def _models_root() -> Path:
    return _root() / 'models'


def _normalize_repo_id(value: str) -> str:
    if not isinstance(value, str):
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, 'model name must be a string')
    value = value.strip().rstrip('/')
    for prefix in ('https://huggingface.co/', 'http://huggingface.co/'):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    if value.count('/') != 1:
        _raise(
            UNIRT_ERROR_COMMON_INVALID_INPUT,
            f"model name must be a Hugging Face 'owner/repository' id, got {value!r}",
        )
    owner, repo = value.split('/', 1)
    if not _REPO_COMPONENT_RE.fullmatch(owner) or not _REPO_COMPONENT_RE.fullmatch(repo):
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, f'invalid model id: {value!r}')
    if '..' in owner or '..' in repo or '--' in owner or '--' in repo:
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, f'invalid model id: {value!r}')
    return value


def _safe_relative_name(name: str) -> str:
    """Validate a Hub/manifest file name for every supported host OS."""

    if not isinstance(name, str) or not name or '\x00' in name or '\\' in name or ':' in name:
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'unsafe model file name: {name!r}')
    if name.startswith('/') or any(part in {'', '.', '..'} for part in name.split('/')):
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'unsafe model file name: {name!r}')
    return PurePosixPath(name).as_posix()


def _cache_file(directory: Path, name: str) -> Path:
    relative = PurePosixPath(_safe_relative_name(name))
    root = directory.resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    if root not in candidate.parents:
        _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'cached file path escapes model directory: {name}')
    return candidate


def _split_precision(value: str) -> tuple[str, str | None]:
    value = value.rstrip('/')
    tail = value.rsplit('/', 1)[-1]
    if ':' not in tail:
        return value, None
    repo, precision = value.rsplit(':', 1)
    return repo, precision or None


def _resolve_name(value: str, precision: str | None) -> tuple[str, str | None]:
    name, embedded = _split_precision(value)
    if '/' not in name:
        resolved = _ALIASES.get(name.lower())
        if resolved:
            name, alias_precision = _split_precision(resolved)
            embedded = embedded or alias_precision
    return _normalize_repo_id(name), precision or embedded


def _model_dir(repo_id: str) -> Path:
    repo_id = _normalize_repo_id(repo_id)
    owner, repo = repo_id.split('/', 1)
    return _models_root() / owner / repo


def _manifest_path(repo_id: str) -> Path:
    return _model_dir(repo_id) / _MANIFEST


def _read_manifest(repo_id: str) -> dict:
    path = _manifest_path(repo_id)
    try:
        if path.stat().st_size > 1024 * 1024:
            _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'manifest is too large: {path}')
        with path.open(encoding='utf-8') as stream:
            value = json.load(stream)
    except FileNotFoundError:
        _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'model is not cached: {repo_id}')
    except (OSError, ValueError) as exc:
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'cannot read {path}: {exc}')
    if not isinstance(value, dict) or value.get('Name') != repo_id:
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid manifest: {path}')
    return value


def _write_manifest(repo_id: str, value: dict) -> None:
    directory = _model_dir(repo_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / _MANIFEST
    temporary = directory / f'.{_MANIFEST}.{os.getpid()}.{threading.get_ident()}.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, target)


def _file_info(
    item: _RemoteFile,
    *,
    files: Sequence[str] | None = None,
    file_sizes: dict[str, int] | None = None,
) -> dict:
    result = {'Name': item.name, 'Downloaded': True, 'Size': max(item.size, 0)}
    if files:
        result['Files'] = list(files)
    if file_sizes:
        result['FileSizes'] = {name: max(int(size), 0) for name, size in file_sizes.items()}
    return result


def _empty_file_info() -> dict:
    return {'Name': '', 'Downloaded': False, 'Size': 0}


def _extract_quant(name: str) -> str | None:
    if _is_mmproj(name):
        return None
    return _extract_quant_token(name)


def _extract_quant_token(name: str) -> str | None:
    matches = [match.group(1).upper() for match in _QUANT_RE.finditer(name.upper())]
    if not matches:
        return None
    for preferred in _QUANT_PRIORITY:
        if preferred in matches:
            return preferred
    return matches[0]


def _select_mmproj(files: Sequence[_RemoteFile]) -> _RemoteFile | None:
    """Prefer Q8_0 projectors, then the smallest deterministic candidate."""
    if not files:
        return None
    q8 = [item for item in files if _extract_quant_token(item.name) == 'Q8_0']
    candidates = q8 or list(files)
    return min(
        candidates,
        key=lambda item: (item.size <= 0, item.size if item.size > 0 else 0, item.name),
    )


def _is_mmproj(name: str) -> bool:
    stem = PurePosixPath(name).name.lower()
    if stem.endswith('.gguf'):
        stem = stem[:-5]
    return (
        stem == 'mmproj'
        or stem.startswith(('mmproj-', 'mmproj.', 'mmproj_'))
        or stem.endswith(('-mmproj', '_mmproj'))
        or '-mmproj-' in stem
        or '_mmproj_' in stem
    )


def _default_precision(values: Iterable[str]) -> str:
    available = set(values)
    for preferred in _QUANT_PRIORITY:
        if preferred in available:
            return preferred
    return sorted(available)[0]


def _model_type(file_names: Sequence[str], config: dict | None = None) -> str:
    if config:
        model_type = str(config.get('model_type') or '').lower()
        if model_type == 'gemma3_text':
            return 'llm'
        if isinstance(config.get('vision_config'), dict) or config.get('mm_vision_tower') is not None:
            return 'vlm'
        architectures = config.get('architectures') or []
        for architecture in architectures:
            lower = str(architecture).lower()
            if lower.endswith('forconditionalgeneration') and any(
                token in lower for token in _VLM_ARCH_TOKENS
            ):
                return 'vlm'
            if lower.endswith('forcausallm'):
                return 'llm'
    return 'vlm' if any(_is_mmproj(name) for name in file_names) else 'llm'


def _derive_model_name(repo_id: str) -> str:
    name = repo_id.rsplit('/', 1)[-1]
    return re.sub(r'-gguf$', '', name, flags=re.IGNORECASE)


def _onnx_precision(name: str) -> str:
    """Return a stable cache key for a model*.onnx variant."""
    stem = PurePosixPath(name).stem.casefold()
    if stem in {'model', 'model-fp32', 'model_fp32'}:
        return 'onnx'
    for prefix in ('model-', 'model_'):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    normalized = re.sub(r'[^a-z0-9]+', '-', stem).strip('-') or 'default'
    return f'onnx-{normalized}'


def _requested_onnx_precision(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r'[^a-z0-9]+', '-', value.casefold()).strip('-')
    if normalized in {'onnx', 'onnx-fp32'}:
        return 'onnx'
    return normalized if normalized.startswith('onnx-') else None


def _validate_numbered_shards(files: Sequence[_RemoteFile], label: str) -> None:
    numbered: list[tuple[int, int, str]] = []
    for item in files:
        match = _NUMBERED_SHARD_RE.search(PurePosixPath(item.name).name)
        if match:
            numbered.append((int(match.group(1)), int(match.group(2)), item.name))
    if not numbered:
        if len(files) > 1:
            _raise(
                UNIRT_ERROR_COMMON_MANIFEST_PARSE,
                f'{label} has multiple unnumbered model files and cannot be dispatched unambiguously',
            )
        return
    if len(numbered) != len(files):
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'{label} mixes numbered and unnumbered model files')
    totals = {total for _, total, _ in numbered}
    if len(totals) != 1:
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'{label} shards disagree on total shard count')
    total = totals.pop()
    indexes = {index for index, _, _ in numbered}
    if total <= 0 or indexes != set(range(1, total + 1)) or len(numbered) != total:
        _raise(
            UNIRT_ERROR_COMMON_MANIFEST_PARSE,
            f'{label} shard set is incomplete: found {sorted(indexes)}, expected 1..{total}',
        )


def _remote_files(repo_id: str, token: str | None) -> list[_RemoteFile]:
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status in (401, 403):
            _raise(UNIRT_ERROR_COMMON_AUTH, f'cannot access {repo_id}: {exc}')
        if status == 404:
            _raise(UNIRT_ERROR_COMMON_HUB_MODEL_NOT_FOUND, f'model not found: {repo_id}')
        _raise(UNIRT_ERROR_COMMON_NETWORK, f'Hugging Face request failed for {repo_id}: {exc}')
    except (OSError, RuntimeError) as exc:
        _raise(UNIRT_ERROR_COMMON_NETWORK, f'Hugging Face request failed for {repo_id}: {exc}')
    return [
        _RemoteFile(_safe_relative_name(str(sibling.rfilename)), int(sibling.size or 0))
        for sibling in (info.siblings or [])
        if sibling.rfilename
    ]


def _load_local_config(directory: Path) -> dict | None:
    try:
        with (directory / 'config.json').open(encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _plan(repo_id: str, files: Sequence[_RemoteFile], precision: str | None) -> _Plan:
    normalized_files = [
        _RemoteFile(_safe_relative_name(item.name), int(item.size))
        for item in files
    ]
    if len({item.name for item in normalized_files}) != len(normalized_files):
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'{repo_id} contains duplicate model file names')
    files = normalized_files
    names = [item.name for item in files]
    ggufs = [item for item in files if item.name.lower().endswith('.gguf') and not _is_mmproj(item.name)]
    mmprojs = [item for item in files if item.name.lower().endswith('.gguf') and _is_mmproj(item.name)]
    monolithic_safetensors = [
        item
        for item in files
        if PurePosixPath(item.name).name.lower() == 'model.safetensors'
    ]
    sharded_safetensors = [
        item
        for item in files
        if _HF_SAFETENSOR_SHARD_RE.match(PurePosixPath(item.name).name)
    ]
    onnx_files = [item for item in files if item.name.casefold().endswith('.onnx')]
    if monolithic_safetensors:
        # A repository can contain checkpoints under subdirectories. Choose
        # only the shallow canonical model instead of merging unrelated files.
        safetensors = [min(monolithic_safetensors, key=lambda item: (item.name.count('/'), item.name))]
    else:
        safetensors = sharded_safetensors

    config = None
    # Remote config contents are intentionally not fetched during query. The
    # filename/mmproj signal is sufficient there; pull reclassifies locally.
    inferred_type = _model_type(names, config)
    model_name = _derive_model_name(repo_id)
    requested_onnx = _requested_onnx_precision(precision)

    if ggufs and requested_onnx is None:
        groups: dict[str, list[_RemoteFile]] = {}
        for item in ggufs:
            groups.setdefault(_extract_quant(item.name) or 'default', []).append(item)
        wanted = precision or _default_precision(groups)
        wanted_key = next((key for key in groups if key.casefold() == wanted.casefold()), None)
        if wanted_key is None:
            available = ', '.join(sorted(groups))
            _raise(
                UNIRT_ERROR_COMMON_INVALID_INPUT,
                f"quantization {wanted!r} not found for {repo_id}; available: {available}",
            )
        model_files = tuple(sorted(groups[wanted_key], key=lambda item: item.name))
        _validate_numbered_shards(model_files, 'GGUF')
        candidates = tuple(
            PrecisionVariant(
                precision=key,
                size=sum(max(item.size, 0) for item in group),
                is_default=key == _default_precision(groups),
            )
            for key, group in sorted(groups.items())
        )
        # Projector precision is independent of the text GGUF. Q8_0 is the
        # practical llama.cpp default and avoids an unnecessary F16 download.
        mmproj = _select_mmproj(mmprojs)
        tokenizer = min(
            (item for item in files if item.name.lower().endswith('tokenizer.json')),
            key=lambda item: (item.name.count('/'), item.name),
            default=None,
        )
        shared = tuple(
            item
            for item in files
            if item.name.lower().endswith(_SIDECAR_SUFFIXES)
            and item not in model_files
            and item != tokenizer
        )
        return _Plan(
            repo_id,
            model_name,
            'llama_cpp',
            inferred_type,
            wanted_key,
            model_files,
            shared,
            mmproj,
            tokenizer,
            candidates,
        )

    if onnx_files and (requested_onnx is not None or not safetensors):
        groups: dict[str, list[_RemoteFile]] = {}
        for item in onnx_files:
            groups.setdefault(_onnx_precision(item.name), []).append(item)
        selected_by_precision = {
            key: min(group, key=lambda item: (item.name.count('/'), len(item.name), item.name))
            for key, group in groups.items()
        }
        wanted = requested_onnx or (
            'onnx' if 'onnx' in selected_by_precision else sorted(selected_by_precision)[0]
        )
        if wanted not in selected_by_precision:
            available = ', '.join(sorted(selected_by_precision))
            _raise(
                UNIRT_ERROR_COMMON_INVALID_INPUT,
                f"ONNX precision {precision!r} not found for {repo_id}; available: {available}",
            )
        selected = selected_by_precision[wanted]
        parent = PurePosixPath(selected.name).parent
        external_data = tuple(
            sorted(
                (
                    item for item in files
                    if PurePosixPath(item.name).parent == parent
                    and item.name != selected.name
                    and item.name.casefold().endswith(('.onnx_data', '.onnx.data'))
                ),
                key=lambda item: item.name,
            )
        )
        model_files = (selected, *external_data)
        tokenizer = min(
            (item for item in files if item.name.casefold().endswith('tokenizer.json')),
            key=lambda item: (item.name.count('/'), item.name),
            default=None,
        )
        shared = tuple(
            item
            for item in files
            if item.name.casefold().endswith(_SIDECAR_SUFFIXES)
            and item not in model_files
            and item != tokenizer
        )
        candidates = tuple(
            PrecisionVariant(
                key,
                max(selected_by_precision[key].size, 0),
                key == ('onnx' if 'onnx' in selected_by_precision else sorted(selected_by_precision)[0]),
            )
            for key in sorted(selected_by_precision)
        )
        return _Plan(
            repo_id,
            model_name,
            'onnxruntime',
            'embedding',
            wanted,
            model_files,
            shared,
            None,
            tokenizer,
            candidates,
        )

    if safetensors:
        if precision and precision.casefold() not in {'default', 'safetensors'}:
            _raise(
                UNIRT_ERROR_COMMON_INVALID_INPUT,
                f"precision {precision!r} is not valid for safetensors repository {repo_id}",
            )
        model_files = tuple(sorted(safetensors, key=lambda item: item.name))
        _validate_numbered_shards(model_files, 'safetensors')
        tokenizer = min(
            (item for item in files if item.name.lower().endswith('tokenizer.json')),
            key=lambda item: (item.name.count('/'), item.name),
            default=None,
        )
        shared = tuple(
            item
            for item in files
            if item.name.lower().endswith((*_SIDECAR_SUFFIXES, '.npy'))
            and item != tokenizer
        )
        total = sum(max(item.size, 0) for item in model_files)
        return _Plan(
            repo_id,
            model_name,
            'mlx',
            inferred_type,
            'default',
            model_files,
            shared,
            None,
            tokenizer,
            (PrecisionVariant('default', total, True),),
        )

    _raise(
        UNIRT_ERROR_COMMON_NOT_SUPPORTED,
        f'{repo_id} has no GGUF, safetensors, or ONNX model files supported by UniRT',
    )


def _manifest_from_plan(plan: _Plan, directory: Path, previous: dict | None = None) -> dict:
    actual = {
        item.name: _RemoteFile(item.name, _cache_file(directory, item.name).stat().st_size)
        for item in plan.files
        if _cache_file(directory, item.name).is_file()
    }
    if len(actual) != len(plan.files):
        missing = sorted(item.name for item in plan.files if item.name not in actual)
        _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'download completed with missing files: {missing}')
    corrupt = sorted(
        item.name
        for item in plan.files
        if item.size > 0 and actual[item.name].size != item.size
    )
    if corrupt:
        _raise(
            UNIRT_ERROR_COMMON_FILE_NOT_FOUND,
            f'download completed with size-mismatched files: {corrupt}',
        )

    local_config = _load_local_config(directory)
    model_type = (
        'embedding'
        if plan.runtime == 'onnxruntime'
        else _model_type(list(actual), local_config)
    )

    entry = actual[plan.model_files[0].name]
    model_map = dict((previous or {}).get('ModelFile') or {})
    previous_plugin = (previous or {}).get('PluginId')
    previous_model_type = (previous or {}).get('ModelType')
    for record in model_map.values():
        if isinstance(record, dict):
            if previous_plugin:
                record.setdefault('PluginId', previous_plugin)
            if previous_model_type:
                record.setdefault('ModelType', previous_model_type)
    model_record = _file_info(
        entry,
        files=[item.name for item in plan.model_files],
        file_sizes={item.name: actual[item.name].size for item in plan.model_files},
    )
    model_record['PluginId'] = plan.runtime
    model_record['ModelType'] = model_type
    model_map[plan.precision] = model_record
    extras = {
        item.get('Name'): item
        for item in ((previous or {}).get('ExtraFiles') or [])
        if isinstance(item, dict)
        and isinstance(item.get('Name'), str)
        and item.get('Name') in actual
    }
    reserved = {item.name for item in plan.model_files}
    if plan.mmproj:
        reserved.add(plan.mmproj.name)
    if plan.tokenizer:
        reserved.add(plan.tokenizer.name)
    for name, item in actual.items():
        if name not in reserved:
            extras[name] = _file_info(item)

    mmproj_info = _file_info(actual[plan.mmproj.name]) if plan.mmproj else _empty_file_info()
    tokenizer_info = _file_info(actual[plan.tokenizer.name]) if plan.tokenizer else _empty_file_info()
    return {
        'Name': plan.repo_id,
        'ModelName': plan.model_name,
        'ModelType': model_type,
        'PluginId': plan.runtime,
        'Precision': plan.precision,
        'ModelFile': model_map,
        'MMProjFile': mmproj_info,
        'TokenizerFile': tokenizer_info,
        'ExtraFiles': [extras[key] for key in sorted(extras)],
    }


class _DownloadCancelled(RuntimeError):
    pass


def _tqdm_for(callback: ProgressCallback | None):
    if callback is None:
        class _SilentTqdm(hub_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs['disable'] = True
                super().__init__(*args, **kwargs)

        return _SilentTqdm

    class _CallbackTqdm(hub_tqdm):
        _active: dict[int, DownloadProgress] = {}
        _cancelled = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._track_bytes = kwargs.get('unit') == 'B'
            self._progress_key = id(self)
            if self._track_bytes:
                self._publish()

        def update(self, n=1):
            result = super().update(n)
            if self._track_bytes:
                self._publish()
            return result

        def close(self):
            try:
                super().close()
            finally:
                if getattr(self, '_track_bytes', False):
                    with _progress_lock:
                        self._active.pop(self._progress_key, None)

        def _publish(self):
            with _progress_lock:
                if self._cancelled:
                    raise _DownloadCancelled('download cancelled by callback')
                name = str(getattr(self, 'desc', '') or 'download').strip()
                total = int(self.total) if self.total is not None else -1
                self._active[self._progress_key] = DownloadProgress(name, int(self.n), total)
                if not callback(list(self._active.values())):
                    type(self)._cancelled = True
                    raise _DownloadCancelled('download cancelled by callback')

    return _CallbackTqdm


def _copy_local(source: Path, destination: Path, callback: ProgressCallback | None) -> None:
    source = source.resolve()
    if not source.is_dir():
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, f'local_path must be a directory: {source}')
    destination = destination.resolve()
    if destination != source and source in destination.parents:
        _raise(
            UNIRT_ERROR_COMMON_INVALID_INPUT,
            'the model cache destination cannot be nested inside local_path',
        )
    for item in sorted(source.rglob('*')):
        relative = item.relative_to(source)
        if not item.is_file() or item.name == _MANIFEST or relative.parts[:2] == ('.cache', 'huggingface'):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = item.stat().st_size
        if callback and not callback([DownloadProgress(relative.as_posix(), 0, size)]):
            _raise(UNIRT_ERROR_COMMON_CANCELLED, 'copy cancelled by callback')
        if item.resolve() != target.resolve():
            shutil.copy2(item, target)
        if callback and not callback([DownloadProgress(relative.as_posix(), size, size)]):
            _raise(UNIRT_ERROR_COMMON_CANCELLED, 'copy cancelled by callback')


def pull(
    model_name: str,
    *,
    precision: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    model_type: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Download one model variant into the UniRT store.

    ``hub`` accepts ``"auto"``/``"hf"`` and ``"localfs"``; anything else
    raises a focused not-supported error.
    """

    repo_id, wanted = _resolve_name(model_name, precision)
    hub_value = _resolve_hub(hub)
    if hub_value == UNIRT_HUB_LOCALFS and not local_path:
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, 'local_path is required for hub="localfs"')
    if hub_value != UNIRT_HUB_LOCALFS and local_path:
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, 'local_path is only valid for hub="localfs"')
    if model_type is not None and model_type.lower() not in {'llm', 'vlm', 'embedding'}:
        _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, f'unknown model type: {model_type!r}')

    destination = _model_dir(repo_id)
    destination.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.environ.get('UNIRT_HFTOKEN') or None

    with _state_lock:
        if local_path:
            _copy_local(Path(local_path), destination, on_progress)
            local_files = [
                _RemoteFile(path.relative_to(destination).as_posix(), path.stat().st_size)
                for path in destination.rglob('*')
                if path.is_file()
                and path.name != _MANIFEST
                and path.relative_to(destination).parts[:2] != ('.cache', 'huggingface')
            ]
            plan = _plan(repo_id, local_files, wanted)
        else:
            remote = _remote_files(repo_id, token)
            plan = _plan(repo_id, remote, wanted)
            try:
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    local_dir=destination,
                    allow_patterns=[item.name for item in plan.files],
                    tqdm_class=_tqdm_for(on_progress),
                )
            except _DownloadCancelled:
                _raise(UNIRT_ERROR_COMMON_CANCELLED, f'download cancelled: {repo_id}')
            except HfHubHTTPError as exc:
                _raise(UNIRT_ERROR_COMMON_NETWORK, f'download failed for {repo_id}: {exc}')
            except (OSError, RuntimeError, ValueError) as exc:
                _raise(UNIRT_ERROR_COMMON_NETWORK, f'download failed for {repo_id}: {exc}')

        previous = None
        try:
            previous = _read_manifest(repo_id)
        except UniRTError as exc:
            if exc.code != UNIRT_ERROR_COMMON_FILE_NOT_FOUND:
                raise
            _set_error(None)
        manifest = _manifest_from_plan(plan, destination, previous)
        if model_type:
            manifest['ModelType'] = model_type.lower()
        _write_manifest(repo_id, manifest)
        _set_error(None)


def query(
    model_name: str,
    *,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
) -> ModelInspection:
    """Inspect available model variants without downloading model bytes."""

    repo_id, precision = _resolve_name(model_name, None)
    hub_value = _resolve_hub(hub)
    if hub_value == UNIRT_HUB_LOCALFS:
        if not local_path:
            _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, 'local_path is required for hub="localfs"')
        directory = Path(local_path).resolve()
        if not directory.is_dir():
            _raise(UNIRT_ERROR_COMMON_INVALID_INPUT, f'local_path must be a directory: {directory}')
        files = [
            _RemoteFile(path.relative_to(directory).as_posix(), path.stat().st_size)
            for path in directory.rglob('*')
            if path.is_file() and path.name != _MANIFEST
        ]
    else:
        token = hf_token or os.environ.get('UNIRT_HFTOKEN') or None
        files = _remote_files(repo_id, token)
    plan = _plan(repo_id, files, precision)
    return ModelInspection(plan.model_name, plan.runtime, plan.model_type, list(plan.candidates))


def _resolve_hub(hub: str | int) -> int:
    values = {
        'auto': UNIRT_HUB_AUTO,
        'hf': UNIRT_HUB_HUGGINGFACE,
        'huggingface': UNIRT_HUB_HUGGINGFACE,
        'local': UNIRT_HUB_LOCALFS,
        'localfs': UNIRT_HUB_LOCALFS,
    }
    if isinstance(hub, int):
        if hub in {UNIRT_HUB_AUTO, UNIRT_HUB_HUGGINGFACE, UNIRT_HUB_LOCALFS}:
            return hub
    else:
        try:
            return values[hub.lower()]
        except (AttributeError, KeyError):
            pass
    _raise(
        UNIRT_ERROR_COMMON_NOT_SUPPORTED,
        f'hub {hub!r} is not supported: only Hugging Face ("hf"/"auto") and '
        'the local filesystem ("localfs") are available',
    )
    raise AssertionError('unreachable')


def list_models() -> list[str]:
    """Return the ids of all valid cached models."""

    return [detail.name for detail in list_detailed()]


def _manifest_total_size(manifest: dict, directory: Path) -> int:
    names: set[str] = set()
    for item in (manifest.get('ModelFile') or {}).values():
        names.update(item.get('Files') or [item.get('Name')])
    for key in ('MMProjFile', 'TokenizerFile'):
        names.add((manifest.get(key) or {}).get('Name'))
    names.update(item.get('Name') for item in (manifest.get('ExtraFiles') or []))
    total = 0
    for name in names:
        if not name:
            continue
        total += _cache_file(directory, name).stat().st_size
    return total


def list_detailed() -> list[CachedModel]:
    """List cached models, skipping incomplete or corrupt entries."""

    result: list[CachedModel] = []
    root = _models_root()
    for path in sorted(root.glob(f'*/*/{_MANIFEST}')):
        try:
            relative = path.relative_to(root)
            repo_id = f'{relative.parts[0]}/{relative.parts[1]}'
            manifest = _read_manifest(repo_id)
            # Reuse the strict resolver so truncated shards and unsafe paths do
            # not appear as healthy cached models.
            healthy_precisions = []
            for precision in sorted((manifest.get('ModelFile') or {}).keys()):
                try:
                    get_paths(f'{repo_id}:{precision}')
                    healthy_precisions.append(precision)
                except UniRTError:
                    continue
            if not healthy_precisions:
                continue
            result.append(
                CachedModel(
                    name=repo_id,
                    model_name=str(manifest.get('ModelName') or _derive_model_name(repo_id)),
                    runtime=str(manifest.get('PluginId') or ''),
                    model_type=str(manifest.get('ModelType') or 'llm'),
                    total_size=_manifest_total_size(manifest, path.parent),
                    precisions=healthy_precisions,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, UniRTError):
            continue
    _set_error(None)
    return result


def get_paths(model_name: str) -> ModelFiles:
    """Resolve a cached ``owner/repo[:precision]`` to local file paths."""

    repo_id, precision = _resolve_name(model_name, None)
    manifest = _read_manifest(repo_id)
    variants = manifest.get('ModelFile') or {}
    if not variants:
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'no model files recorded for {repo_id}')
    if precision:
        key = next((value for value in variants if value.casefold() == precision.casefold()), None)
        if key is None:
            _raise(
                UNIRT_ERROR_COMMON_FILE_NOT_FOUND,
                f"precision {precision!r} is not cached for {repo_id}",
            )
    else:
        top = str(manifest.get('Precision') or '')
        key = top if top in variants else _default_precision(variants)

    directory = _model_dir(repo_id).resolve()

    def resolve_name(name: str, expected_size: int | None = None) -> str:
        if not name:
            _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'empty cached file name for {repo_id}')
        candidate = _cache_file(directory, name)
        if not candidate.is_file():
            _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'cached file is missing or unsafe: {name}')
        if expected_size is not None and expected_size > 0 and candidate.stat().st_size != expected_size:
            _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'cached file has wrong size: {name}')
        return str(candidate)

    def resolve_record(record: dict | None, *, validate_all: bool = False) -> str | None:
        value = record or {}
        name = str(value.get('Name') or '')
        if not name:
            return None
        sizes = value.get('FileSizes') if isinstance(value.get('FileSizes'), dict) else {}
        if validate_all:
            names = value.get('Files') or [name]
            if not isinstance(names, list) or name not in names:
                _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid model file list for {repo_id}')
            for item in names:
                if not isinstance(item, str):
                    _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid cached file name for {repo_id}')
                expected = sizes.get(item)
                resolve_name(item, int(expected) if isinstance(expected, int) else None)
        expected = sizes.get(name, value.get('Size'))
        return resolve_name(name, int(expected) if isinstance(expected, int) else None)

    selected_record = variants[key]
    if not isinstance(selected_record, dict):
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid model record for {repo_id}:{key}')
    model_path = resolve_record(selected_record, validate_all=True) or ''
    mmproj_path = resolve_record(manifest.get('MMProjFile'))
    tokenizer_path = resolve_record(manifest.get('TokenizerFile'))
    extras = manifest.get('ExtraFiles') or []
    if not isinstance(extras, list):
        _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid extra file list for {repo_id}')
    for record in extras:
        if not isinstance(record, dict):
            _raise(UNIRT_ERROR_COMMON_MANIFEST_PARSE, f'invalid extra file record for {repo_id}')
        resolve_record(record)

    return ModelFiles(
        model_path=model_path,
        model_dir=str(directory),
        model_name=str(manifest.get('ModelName') or _derive_model_name(repo_id)),
        runtime=str(selected_record.get('PluginId') or manifest.get('PluginId') or ''),
        model_type=str(selected_record.get('ModelType') or manifest.get('ModelType') or 'llm'),
        mmproj_path=mmproj_path,
        tokenizer_path=tokenizer_path,
    )


def get_type(model_name: str) -> str:
    repo_id, _ = _resolve_name(model_name, None)
    return str(_read_manifest(repo_id).get('ModelType') or 'llm')


def set_type(model_name: str, model_type: str) -> None:
    repo_id, _ = _resolve_name(model_name, None)
    model_type = model_type.lower()
    if model_type not in {'llm', 'vlm'}:
        raise ValueError(f"unknown model type: {model_type!r} (expected 'llm' or 'vlm')")
    with _state_lock:
        manifest = _read_manifest(repo_id)
        manifest['ModelType'] = model_type
        _write_manifest(repo_id, manifest)


def remove(model_name: str) -> None:
    """Remove a cached repository, or one ``:precision`` variant."""

    repo_id, precision = _resolve_name(model_name, None)
    directory = _model_dir(repo_id)
    with _state_lock:
        if not directory.exists():
            _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'model is not cached: {repo_id}')
        if precision is None:
            shutil.rmtree(directory)
            return
        manifest = _read_manifest(repo_id)
        variants = manifest.get('ModelFile') or {}
        key = next((value for value in variants if value.casefold() == precision.casefold()), None)
        if key is None:
            _raise(UNIRT_ERROR_COMMON_FILE_NOT_FOUND, f'precision is not cached: {precision}')
        if len(variants) == 1:
            shutil.rmtree(directory)
            return
        entry = variants.pop(key)
        for name in entry.get('Files') or [entry.get('Name')]:
            if name:
                try:
                    target = _cache_file(directory, name)
                    target.unlink()
                except FileNotFoundError:
                    pass
        manifest['Precision'] = _default_precision(variants)
        _write_manifest(repo_id, manifest)


def clean() -> int:
    """Remove every cached model and return the number removed."""

    names = list_models()
    for name in names:
        remove(name)
    return len(names)




def ensure_cached(
    model_name_or_alias: str,
    *,
    precision: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> ModelFiles:
    """Return a local model, downloading the requested variant if necessary."""

    repo_id, wanted = _resolve_name(model_name_or_alias, precision)
    key = f'{repo_id}:{wanted}' if wanted else repo_id
    try:
        return get_paths(key)
    except UniRTError as exc:
        if exc.code != UNIRT_ERROR_COMMON_FILE_NOT_FOUND:
            raise
        _set_error(None)
    if wanted is None and _resolve_hub(hub) != UNIRT_HUB_LOCALFS:
        result = query(repo_id, hub=hub, hf_token=hf_token)
        wanted = next((item.precision for item in result.candidates if item.is_default), None)
    pull(
        repo_id,
        precision=wanted,
        hub=hub,
        local_path=local_path,
        hf_token=hf_token,
        on_progress=on_progress,
    )
    return get_paths(f'{repo_id}:{wanted}' if wanted else repo_id)

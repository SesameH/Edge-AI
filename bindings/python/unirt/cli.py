# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line model management and interactive chat."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time

import unirt
from unirt import (
    AutoModelForCausalLM,
    AutoModelForEmbedding,
    UniRTError,
    UniRTVLM,
    _progress,
    get_compute_unit_list,
    get_runtime_list,
    init,
    set_log_level,
    version,
)

_models = unirt.model_manager


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


_MEDIA_PATH = re.compile(
    r'(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|mp3|wav)\b'
)
_IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp'})
_AUDIO_EXTENSIONS = frozenset({'.mp3', '.wav'})


def _parse_media(prompt: str) -> tuple[str, list[str], list[str]]:
    images: list[str] = []
    audios: list[str] = []
    cleaned = prompt
    for match in _MEDIA_PATH.findall(prompt):
        if not os.path.isfile(match):
            print(f'warning: file not found: {match}', file=sys.stderr)
            continue
        extension = os.path.splitext(match)[1].lower()
        if extension in _IMAGE_EXTENSIONS:
            images.append(match)
        elif extension in _AUDIO_EXTENSIONS:
            audios.append(match)
        cleaned = cleaned.replace(f"'{match}'", '').replace(match, '')
    return cleaned.strip(), images, audios


def _collect_media_history(messages: list[dict]) -> tuple[list[str], list[str]]:
    """Return per-modality paths in the same order as full-history markers."""
    images: list[str] = []
    audios: list[str] = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'image' and isinstance(block.get('image'), str):
                images.append(block['image'])
            elif block.get('type') == 'audio' and isinstance(block.get('audio'), str):
                audios.append(block['audio'])
    return images, audios


def _ensure_downloaded(
    model: str,
    quant: str | None,
    *,
    hub: str = 'auto',
    local_path: str | None = None,
) -> _models.ModelFiles | None:
    if os.path.exists(model):
        return None
    local_hub = hub in {'localfs', 'local'}
    if local_hub and not local_path:
        raise ValueError('--local-path is required when --hub localfs')

    progress = _progress.default_progress_printer()
    outcome: dict[str, object] = {}

    def download() -> None:
        try:
            if local_hub:
                _models.pull(
                    model,
                    precision=quant,
                    hub=hub,
                    local_path=local_path,
                    hf_token=os.environ.get('UNIRT_HFTOKEN'),
                    on_progress=progress,
                )
                key = f'{model}:{quant}' if quant else model
                outcome['paths'] = _models.get_paths(key)
            else:
                outcome['paths'] = _models.ensure_cached(
                    model,
                    precision=quant,
                    hub=hub,
                    hf_token=os.environ.get('UNIRT_HFTOKEN'),
                    on_progress=progress,
                )
        except BaseException as exc:
            outcome['error'] = exc

    worker = threading.Thread(target=download, name='unirt-download', daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(timeout=0.1)
    except KeyboardInterrupt:
        _progress.finish(progress)
        sys.stderr.write('(aborted — partial download preserved; rerun to resume)\n')
        sys.stderr.flush()
        os._exit(130)
    _progress.finish(progress)

    error = outcome.get('error')
    if isinstance(error, BaseException):
        raise error
    paths = outcome.get('paths')
    return paths if isinstance(paths, _models.ModelFiles) else None


_USE_COLOR = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None
_DIM = '\x1b[2m' if _USE_COLOR else ''
_CYAN = '\x1b[36m' if _USE_COLOR else ''
_GREEN = '\x1b[32m' if _USE_COLOR else ''
_RESET = '\x1b[0m' if _USE_COLOR else ''


def _typed_user_content(
    text: str,
    images: list[str],
    audios: list[str],
) -> list[dict] | str:
    blocks: list[dict] = [
        *({'type': 'image', 'image': path} for path in images),
        *({'type': 'audio', 'audio': path} for path in audios),
    ]
    if text:
        blocks.append({'type': 'text', 'text': text})
    return blocks or text


def _check_requested_modalities(
    images: list[str],
    audios: list[str],
    capabilities: dict[str, bool] | None,
) -> None:
    if capabilities is None:
        return
    if images and not capabilities.get('vision', False):
        raise ValueError('the loaded VLM does not support image input')
    if audios and not capabilities.get('audio', False):
        raise ValueError('the loaded VLM does not support audio input')


def _run_turn(
    model,
    history: list[dict],
    user: str,
    max_tokens: int,
    temperature: float,
    *,
    is_vlm: bool,
    capabilities: dict[str, bool] | None = None,
) -> None:
    images: list[str] = []
    audios: list[str] = []
    if is_vlm:
        text, new_images, new_audios = _parse_media(user)
        _check_requested_modalities(new_images, new_audios, capabilities)
        history.append({
            'role': 'user',
            'content': _typed_user_content(text, new_images, new_audios),
        })
        images, audios = _collect_media_history(history)
    else:
        history.append({'role': 'user', 'content': user})

    prompt = model.tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    arguments: dict = {
        'max_new_tokens': max_tokens,
        'temperature': temperature,
        'stream': True,
    }
    if is_vlm:
        arguments.update(images=images, audios=audios)

    pieces: list[str] = []
    streamer = model.generate(prompt, **arguments)
    try:
        for piece in streamer:
            pieces.append(piece)
            print(piece, end='', flush=True)
        print()
    except KeyboardInterrupt:
        streamer.cancel()
        try:
            for piece in streamer:
                pieces.append(piece)
                print(piece, end='', flush=True)
        except BaseException:
            pass
        print()

    if streamer.output is not None:
        profile = streamer.output.profile
        reason = profile.stop_reason or 'unknown'
        print(
            f'\n{_CYAN}— {profile.decode_speed:.1f} tok/s · '
            f'{profile.generated_tokens} tok · {profile.ttft / 1e6:.1f} s first token '
            f'· stop: {reason} —{_RESET}\n'
        )
    history.append({'role': 'assistant', 'content': ''.join(pieces)})


def _chat_loop(
    model,
    system: str | None,
    max_tokens: int,
    temperature: float,
    *,
    is_vlm: bool,
    capabilities: dict[str, bool] | None = None,
) -> None:
    history: list[dict] = []
    if system:
        history.append({'role': 'system', 'content': system})
    while True:
        try:
            user = input(f'{_GREEN}> {_RESET}')
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            return

        command = user.strip()
        if not command:
            continue
        if command in {'/exit', '/quit'}:
            return
        if command == '/reset':
            history = [history[0]] if system else []
            model.reset()
            print(f'{_DIM}(history cleared){_RESET}')
            continue
        _run_turn(
            model,
            history,
            user,
            max_tokens,
            temperature,
            is_vlm=is_vlm,
            capabilities=capabilities,
        )


_VERBOSITY_LEVELS = {1: 'info', 2: 'debug'}
_LOG_LEVEL_CHOICES = ('trace', 'debug', 'info', 'warn', 'error', 'none')


def _resolve_log_level(arguments: argparse.Namespace) -> str | None:
    if arguments.log_level:
        return arguments.log_level
    if arguments.verbose <= 0:
        return None
    return _VERBOSITY_LEVELS.get(arguments.verbose, 'trace')


def _apply_log_level(level: str) -> None:
    set_log_level(level)
    python_level = {
        'trace': logging.DEBUG,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'none': logging.CRITICAL + 1,
    }[level]
    logger = logging.getLogger('unirt')
    if not any(getattr(handler, '_unirt_cli', False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
        handler._unirt_cli = True
        logger.addHandler(handler)
    logger.setLevel(python_level)
    logger.propagate = False


def _cmd_version(_arguments: argparse.Namespace) -> int:
    init()
    print(f'unirt (python): {unirt.__version__}')
    print(f'SDK:             {version()}')
    return 0


def _cmd_devices(_arguments: argparse.Namespace) -> int:
    init()
    runtimes = get_runtime_list()
    if not runtimes:
        print('No plugins available.')
        return 0
    for runtime in runtimes:
        print(f'{runtime}:')
        devices = get_compute_unit_list(runtime)
        if not devices:
            print('  (no devices)')
        for device_id, device_name in devices:
            print(f'  {device_id:<16} {device_name}')
    return 0


def _cmd_chat(arguments: argparse.Namespace) -> int:
    _ensure_downloaded(
        arguments.model,
        arguments.quant,
        hub=arguments.hub,
        local_path=arguments.local_path,
    )
    display = (
        f'{arguments.model}:{arguments.quant}'
        if arguments.quant else arguments.model
    )
    print(f'{_DIM}loading {display} ...{_RESET} ', end='', flush=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        precision=arguments.quant,
        device_map=arguments.device,
        n_ctx=arguments.n_ctx,
    )
    try:
        is_vlm = isinstance(model, UniRTVLM)
        capabilities = model.capabilities() if is_vlm else None
        meta = getattr(model, '_meta', None) or {}
        runtime = meta.get('backend')
        device = meta.get('device')
        location = f'{runtime}:{device}' if runtime and device else (runtime or arguments.device)
        kind = 'vlm' if is_vlm else 'llm'
        print(
            f'{_DIM}done ({kind}, {time.monotonic() - started:.1f}s, '
            f'{location}){_RESET}'
        )

        if arguments.prompt is not None:
            history: list[dict] = []
            if arguments.system:
                history.append({'role': 'system', 'content': arguments.system})
            _run_turn(
                model,
                history,
                arguments.prompt,
                arguments.max_tokens,
                arguments.temperature,
                is_vlm=is_vlm,
                capabilities=capabilities,
            )
            return 0

        if is_vlm:
            supported = '/'.join(
                name for name, enabled in (capabilities or {}).items() if enabled
            ) or 'no media'
            print(
                f'{_DIM}VLM mode ({supported}) — include png/jpg/webp/mp3/wav '
                f'paths in a prompt to attach media.{_RESET}'
            )
        print(f'{_DIM}Use Ctrl+D or /exit to exit.{_RESET}\n')
        _chat_loop(
            model,
            arguments.system,
            arguments.max_tokens,
            arguments.temperature,
            is_vlm=is_vlm,
            capabilities=capabilities,
        )
        return 0
    finally:
        model.close()


def _cmd_pull(arguments: argparse.Namespace) -> int:
    _ensure_downloaded(
        arguments.model,
        arguments.quant,
        hub=arguments.hub,
        local_path=arguments.local_path,
    )
    return 0


def _cmd_embed(arguments: argparse.Namespace) -> int:
    model = AutoModelForEmbedding.from_pretrained(
        arguments.model,
        precision=arguments.precision,
        device_map=arguments.device,
        pooling=arguments.pooling,
        normalize=not arguments.no_normalize,
    )
    try:
        vectors = model.encode(arguments.texts)
        print(json.dumps(vectors, ensure_ascii=False))
        return 0
    finally:
        model.close()


def _human_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024:
            return f'{int(size)} {unit}' if unit == 'B' else f'{size:.0f} {unit}'
        size /= 1024
    return f'{size:.1f} PiB'


def _read_manifest(model: str) -> dict | None:
    try:
        paths = _models.get_paths(model)
        with open(os.path.join(paths.model_dir, 'unirt.json'), encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (UniRTError, OSError, ValueError):
        return None


def _render_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

    def render(row: list[str]) -> str:
        return '| ' + ' | '.join(
            cell.ljust(width) for cell, width in zip(row, widths)
        ) + ' |'

    print(border)
    print(render(headers))
    print(border)
    for row in rows:
        print(render(row))
    print(border)


def _manifest_file_size(record: dict, model_dir: str | None) -> int:
    if not record.get('Downloaded'):
        return 0
    size = max(int(record.get('Size') or 0), 0)
    if size or not model_dir or not record.get('Name'):
        return size
    try:
        return os.path.getsize(os.path.join(model_dir, record['Name']))
    except OSError:
        return 0


def _cmd_ls(arguments: argparse.Namespace) -> int:
    if arguments.model:
        manifest = _read_manifest(arguments.model)
        if manifest is None:
            print(f'error: {arguments.model} is not cached', file=sys.stderr)
            return 1
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    names = _models.list_models()
    if not names:
        print(f'{_DIM}(no models cached){_RESET}')
        return 0
    rows: list[list[str]] = []
    for name in names:
        manifest = _read_manifest(name) or {}
        try:
            model_dir = _models.get_paths(name).model_dir
        except UniRTError:
            model_dir = None

        variants = manifest.get('ModelFile') or {}
        precisions = sorted(variants)
        top_precision = manifest.get('Precision') or ''
        if top_precision and precisions == ['N/A']:
            precisions = [top_precision]
        records = [
            *variants.values(),
            manifest.get('MMProjFile') or {},
            manifest.get('TokenizerFile') or {},
            *(manifest.get('ExtraFiles') or []),
        ]
        total = sum(_manifest_file_size(record, model_dir) for record in records)
        rows.append([
            name,
            _human_size(total) if total else '?',
            manifest.get('PluginId', '') or '',
            manifest.get('ModelType', '') or '',
            ','.join(precisions) or '-',
        ])
    _render_table(rows, ['NAME', 'SIZE', 'PLUGIN', 'TYPE', 'PRECISIONS'])
    return 0


def _cmd_rm(arguments: argparse.Namespace) -> int:
    if arguments.all:
        count = _models.clean()
        print(f'removed {count} model{"s" if count != 1 else ""}')
        return 0
    if not arguments.model:
        print('error: specify a model name or --all', file=sys.stderr)
        return 2
    _models.remove(arguments.model)
    print(f'removed {arguments.model}')
    return 0


def _add_hub_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--hub',
        choices=['auto', 'hf', 'huggingface', 'localfs', 'local'],
        default='auto',
        help='Source hub (default: auto = HuggingFace)',
    )
    parser.add_argument(
        '--local-path',
        default=None,
        help='Source directory (required when --hub localfs)',
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='unirt-py',
        description='UniRT Python CLI',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Increase log verbosity: -v=info, -vv=debug, -vvv+=trace',
    )
    parser.add_argument(
        '--log-level',
        choices=_LOG_LEVEL_CHOICES,
        default=None,
        help='Set SDK log level explicitly (overrides -v and UNIRT_LOG)',
    )
    commands = parser.add_subparsers(dest='cmd', required=True)

    chat = commands.add_parser('chat', help='Interactive chat with a model')
    chat.add_argument('model', help='Alias, Hugging Face repo id, or local path')
    chat.add_argument('--quant', default=None, help='Quantization variant (e.g. Q4_K_M)')
    chat.add_argument('--system', default=None, help='Optional system prompt')
    chat.add_argument('-p', '--prompt', default=None, help='Run one turn and exit')
    chat.add_argument('--max-tokens', type=int, default=512)
    chat.add_argument('--temperature', type=float, default=0.7)
    chat.add_argument('--n-ctx', type=int, default=0, help='Context length (0 = model default)')
    chat.add_argument(
        '--device',
        default='auto',
        help=(
            "'auto' | 'cpu' | 'gpu' | 'npu' | 'hybrid' | '<plugin>' | "
            "'<plugin>:<device>' (default: auto; run 'unirt-py devices' to list ids)"
        ),
    )
    _add_hub_args(chat)
    chat.set_defaults(func=_cmd_chat)

    pull = commands.add_parser('pull', help='Download a model into the local cache')
    pull.add_argument('model', help='Alias or HF repo id (supports org/repo:precision)')
    pull.add_argument('--quant', default=None, help='Quantization variant (e.g. Q4_K_M)')
    _add_hub_args(pull)
    pull.set_defaults(func=_cmd_pull)

    embed = commands.add_parser('embed', help='Encode text with an ONNX embedding model')
    embed.add_argument('model', help='Hugging Face repo id, ONNX file, or local bundle')
    embed.add_argument('texts', nargs='+', help='One or more strings to encode')
    embed.add_argument('--precision', default=None, help='ONNX variant, e.g. qint8_arm64')
    embed.add_argument('--device', choices=['auto', 'cpu', 'coreml'], default='auto')
    embed.add_argument(
        '--pooling',
        choices=['default', 'cls', 'mean', 'last_token'],
        default='mean',
    )
    embed.add_argument('--no-normalize', action='store_true')
    embed.set_defaults(func=_cmd_embed)

    listing = commands.add_parser('ls', help='List cached models or show one manifest')
    listing.add_argument('model', nargs='?')
    listing.set_defaults(func=_cmd_ls)

    remove = commands.add_parser('rm', help='Remove cached models')
    remove.add_argument('model', nargs='?')
    remove.add_argument('--all', action='store_true')
    remove.set_defaults(func=_cmd_rm)

    devices = commands.add_parser('devices', help='List plugins and devices')
    devices.set_defaults(func=_cmd_devices)
    version_command = commands.add_parser('version', help='Print package and SDK versions')
    version_command.set_defaults(func=_cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    arguments = _build_parser().parse_args(argv)
    level = _resolve_log_level(arguments)
    try:
        if level is not None:
            init()
            _apply_log_level(level)
        return arguments.func(arguments)
    except UniRTError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI-compatible HTTP server over a UniRT model. Stdlib only.

    python3 -m unirt.server --model models/SmolLM2-135M-Instruct --backend mlx --port 8080

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (supports "stream": true via SSE)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import signal
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .auto import AutoModelForCausalLM
from .modeling import UniRTVLM

_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_MEDIA_BYTES = 12 * 1024 * 1024


def _completion_id() -> str:
    return 'chatcmpl-' + uuid.uuid4().hex[:24]


def _with_clean_model_state(model, lock: threading.Lock, operation):
    """Serialize one operation and guarantee clean KV state on both edges."""

    with lock:
        model.reset()
        try:
            return operation()
        finally:
            model.reset()


def _parse_generation_args(req: dict) -> dict:
    max_tokens = req.get('max_tokens', req.get('max_completion_tokens', 512))
    temperature = req.get('temperature', 0.8)
    top_p = req.get('top_p', 0.95)
    seed = req.get('seed', 0)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError('max_tokens must be an integer')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError('temperature must be a number')
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError('top_p must be a number')
    temperature = float(temperature)
    top_p = float(top_p)
    if not 0 < max_tokens <= 2**31 - 1:
        raise ValueError('max_tokens must be positive and fit in int32')
    if not -(2**31) <= seed <= 2**31 - 1:
        raise ValueError('seed must fit in int32')
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be finite and between 0 and 2')
    if not math.isfinite(top_p) or not 0 <= top_p <= 1:
        raise ValueError('top_p must be finite and between 0 and 1')

    result = {
        'max_new_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'seed': seed,
    }
    stop = req.get('stop')
    if isinstance(stop, str):
        if '\x00' in stop:
            raise ValueError('stop must not contain NUL bytes')
        result['stop'] = [stop]
    elif stop is not None:
        if not isinstance(stop, list) or not all(
            isinstance(item, str) and '\x00' not in item for item in stop
        ):
            raise ValueError('stop must be a string or an array of strings')
        result['stop'] = stop
    return result


def _validate_messages(messages) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty array")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each item in 'messages' must be an object")
        role = message.get('role', 'user')
        content = message.get('content', '')
        if content is None:
            content = ''
        if not isinstance(role, str) or not role or '\x00' in role:
            raise ValueError('message role must be a non-empty NUL-free string')
        if not isinstance(content, str) or '\x00' in content:
            raise ValueError(
                'message content must be a NUL-free string; this server build is text-only'
            )
        normalized.append({'role': role, 'content': content})
    return normalized


@dataclass
class _PreparedMessages:
    messages: list[dict]
    images: list[str]
    audios: list[str]
    temporary_paths: list[str]

    def cleanup(self) -> None:
        for path in self.temporary_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _decode_media_data(data: str, suffix: str, temporary_paths: list[str]) -> str:
    try:
        payload = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f'invalid base64 media data: {exc}') from exc
    if not payload or len(payload) > _MAX_MEDIA_BYTES:
        raise ValueError('media payload must be non-empty and at most 12 MiB')
    with tempfile.NamedTemporaryFile(
        prefix='unirt-media-', suffix=suffix, delete=False
    ) as stream:
        stream.write(payload)
        path = stream.name
    temporary_paths.append(path)
    return path


def _prepare_messages(
    messages,
    *,
    multimodal: bool,
    capabilities: dict[str, bool] | None = None,
) -> _PreparedMessages:
    """Validate a request and materialize OpenAI data-URL media safely."""
    if not multimodal:
        return _PreparedMessages(_validate_messages(messages), [], [], [])
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty array")

    normalized: list[dict] = []
    images: list[str] = []
    audios: list[str] = []
    temporary_paths: list[str] = []
    try:
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each item in 'messages' must be an object")
            role = message.get('role', 'user')
            content = message.get('content', '')
            if content is None:
                content = ''
            if not isinstance(role, str) or not role or '\x00' in role:
                raise ValueError('message role must be a non-empty NUL-free string')
            if isinstance(content, str):
                if '\x00' in content:
                    raise ValueError('message content must be NUL-free')
                normalized.append({'role': role, 'content': content})
                continue
            if not isinstance(content, list):
                raise ValueError('multimodal message content must be a string or array')

            parts: list[dict] = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError('multimodal content blocks must be objects')
                kind = item.get('type')
                if kind == 'text':
                    text = item.get('text', '')
                    if not isinstance(text, str) or '\x00' in text:
                        raise ValueError('text blocks must contain NUL-free text')
                    parts.append({'type': 'text', 'text': text})
                    continue
                if kind == 'image_url':
                    if capabilities is not None and not capabilities.get('vision', False):
                        raise ValueError('the loaded VLM does not support image input')
                    image_url = item.get('image_url')
                    url = image_url.get('url') if isinstance(image_url, dict) else image_url
                    if not isinstance(url, str) or not url.startswith('data:image/'):
                        raise ValueError('image_url must be an inline data:image/...;base64 URL')
                    try:
                        header, encoded = url.split(',', 1)
                    except ValueError as exc:
                        raise ValueError('image_url data URL is missing a comma') from exc
                    mime = header[5:].lower()
                    suffixes = {
                        'image/jpeg;base64': '.jpg',
                        'image/png;base64': '.png',
                        'image/webp;base64': '.webp',
                    }
                    suffix = suffixes.get(mime)
                    if suffix is None:
                        raise ValueError('image_url must contain JPEG, PNG, or WebP base64 data')
                    path = _decode_media_data(encoded, suffix, temporary_paths)
                    images.append(path)
                    parts.append({'type': 'image', 'image': path})
                    continue
                if kind == 'input_audio':
                    if capabilities is not None and not capabilities.get('audio', False):
                        raise ValueError('the loaded VLM does not support audio input')
                    audio = item.get('input_audio')
                    if not isinstance(audio, dict):
                        raise ValueError('input_audio must be an object')
                    encoded = audio.get('data')
                    fmt = str(audio.get('format') or '').lower()
                    if not isinstance(encoded, str) or fmt not in {'wav', 'mp3'}:
                        raise ValueError('input_audio requires base64 data and wav/mp3 format')
                    path = _decode_media_data(encoded, f'.{fmt}', temporary_paths)
                    audios.append(path)
                    parts.append({'type': 'audio', 'audio': path})
                    continue
                raise ValueError(f'unsupported multimodal content type: {kind!r}')
            normalized.append({'role': role, 'content': parts})
    except Exception:
        _PreparedMessages([], [], [], temporary_paths).cleanup()
        raise
    return _PreparedMessages(normalized, images, audios, temporary_paths)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # ---- helpers ----

    def _cors(self) -> None:
        # Browser clients (web UIs on another origin) need these on every
        # response; the API carries no cookies, so a wildcard is safe.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def _json(
        self, code: int, payload: dict, *, extra_headers: dict[str, str] | None = None
    ) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._cors()
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._response_started = True
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._json(code, {'error': {'message': message, 'type': 'invalid_request_error'}})

    def _busy(self) -> None:
        # Generation is serialized behind one model/lock; request_slots bounds
        # how many callers may be queued waiting for it so load sheds with a
        # clear signal instead of piling up unbounded blocked threads.
        self._json(
            503,
            {'error': {
                'message': 'server is busy handling other requests; retry shortly',
                'type': 'server_error',
            }},
            extra_headers={'Retry-After': '1'},
        )

    # ---- routes ----

    def do_OPTIONS(self):
        # CORS preflight for browser-based clients.
        self.send_response(204)
        self._cors()
        self.send_header('Access-Control-Max-Age', '86400')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        model_id = self.server.model_id
        if self.path == '/v1/models':
            self._json(200, {
                'object': 'list',
                'data': [{'id': model_id, 'object': 'model', 'owned_by': 'unirt'}],
            })
        elif self.path in ('/', '/health'):
            self._json(200, {'status': 'ok', 'model': model_id})
        else:
            self._error(404, f'unknown path {self.path}')

    def do_POST(self):
        self._response_started = False
        if self.path != '/v1/chat/completions':
            self._error(404, f'unknown path {self.path}')
            return
        try:
            raw_length = self.headers.get('Content-Length')
            if raw_length is None:
                self._error(411, 'Content-Length is required')
                return
            length = int(raw_length)
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                self._error(413 if length > _MAX_REQUEST_BYTES else 400, 'invalid request body size')
                return
            req = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError) as e:
            self._error(400, f'bad request body: {e}')
            return
        if not isinstance(req, dict):
            self._error(400, 'request body must be a JSON object')
            return

        try:
            gen_kwargs = _parse_generation_args(req)
            if not isinstance(req.get('stream', False), bool):
                raise ValueError('stream must be a boolean')
            prepared = _prepare_messages(
                req.get('messages'),
                multimodal=isinstance(self.server.model, UniRTVLM),
                capabilities=self.server.capabilities,
            )
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if not self.server.request_slots.acquire(blocking=False):
            self._busy()
            prepared.cleanup()
            return

        model = self.server.model
        try:
            # Generation mutates KV state; serialize and bracket every request
            # with reset so exceptions/disconnects cannot poison the next one.
            def generate_response():
                prompt = model._apply_chat_template(
                    prepared.messages,
                    True,
                    False,
                    None,
                )
                if isinstance(model, UniRTVLM):
                    gen_kwargs['images'] = prepared.images
                    gen_kwargs['audios'] = prepared.audios
                if req.get('stream'):
                    self._stream_completion(prompt, gen_kwargs)
                else:
                    self._blocking_completion(prompt, gen_kwargs)

            _with_clean_model_state(model, self.server.gen_lock, generate_response)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 — HTTP boundary
            if not self._response_started:
                self._error(500, f'generation failed: {exc}')
            else:
                self.close_connection = True
        finally:
            self.server.request_slots.release()
            prepared.cleanup()

    # ---- completion modes ----

    def _blocking_completion(self, prompt: str, gen_kwargs: dict) -> None:
        out = self.server.model.generate(prompt, **gen_kwargs)
        p = out.profile
        finish = 'stop' if p.stop_reason in ('eos', 'stop_sequence') else 'length'
        self._json(200, {
            'id': _completion_id(),
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': self.server.model_id,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': out.text},
                'finish_reason': finish,
            }],
            'usage': {
                'prompt_tokens': p.prompt_tokens,
                'completion_tokens': p.generated_tokens,
                'total_tokens': p.prompt_tokens + p.generated_tokens,
            },
        })

    def _stream_completion(self, prompt: str, gen_kwargs: dict) -> None:
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        self._response_started = True

        cid = _completion_id()
        created = int(time.time())

        def send_chunk(payload: dict) -> None:
            data = b'data: ' + json.dumps(payload).encode() + b'\n\n'
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
            self.wfile.flush()

        def delta(d: dict, finish=None) -> dict:
            return {
                'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                'model': self.server.model_id,
                'choices': [{'index': 0, 'delta': d, 'finish_reason': finish}],
            }

        send_chunk(delta({'role': 'assistant', 'content': ''}))
        streamer = self.server.model.generate(prompt, stream=True, **gen_kwargs)
        try:
            for piece in streamer:
                send_chunk(delta({'content': piece}))
        except (BrokenPipeError, ConnectionResetError):
            streamer.cancel()
            try:
                for _ in streamer:
                    pass
            except BaseException:
                pass
            raise
        out = streamer.output
        finish = 'stop' if out and out.profile.stop_reason in ('eos', 'stop_sequence') else 'length'
        send_chunk(delta({}, finish=finish))
        done = b'data: [DONE]\n\n'
        self.wfile.write(f'{len(done):x}\r\n'.encode() + done + b'\r\n')
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()


class UniRTHTTPServer(ThreadingHTTPServer):
    """Threading server that owns request state without module globals."""

    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, address, model, model_id: str, max_queued_requests: int = 8):
        self.model = model
        self.model_id = model_id
        self.capabilities = model.capabilities() if isinstance(model, UniRTVLM) else None
        self.gen_lock = threading.Lock()
        # Generation itself is fully serialized by gen_lock (one model, one KV
        # state); this only bounds how many callers may be queued waiting for
        # it, so a burst sheds load with 503s instead of piling up unbounded
        # blocked threads.
        self.request_slots = threading.Semaphore(max_queued_requests)
        super().__init__(address, Handler)


def serve(
    model, model_id: str, host: str, port: int, max_queued_requests: int = 8
) -> None:
    """Serve until interrupted, then stop accepting and join active requests."""

    server = UniRTHTTPServer((host, port), model, model_id, max_queued_requests)
    previous_handlers: dict[int, object] = {}

    def request_shutdown(_signum, _frame):
        # BaseServer.shutdown() must be called from a different thread than
        # serve_forever(), otherwise it deadlocks.
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
        except (ValueError, OSError):
            pass
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description='OpenAI-compatible server over a UniRT model')
    ap.add_argument(
        '--model',
        required=True,
        help='local model path or Hugging Face repository id',
    )
    ap.add_argument('--backend', choices=['llama_cpp', 'mlx'], default='llama_cpp')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument(
        '--n-ctx',
        type=int,
        default=0,
        help='context window in tokens (0 = the model default)',
    )
    ap.add_argument(
        '--max-queued-requests',
        type=int,
        default=8,
        help='requests may queue waiting for the (serialized) model before '
             'the server starts returning 503 (default: 8)',
    )
    args = ap.parse_args()
    if args.n_ctx < 0:
        ap.error('--n-ctx must be >= 0')
    if args.max_queued_requests < 1:
        ap.error('--max-queued-requests must be >= 1')

    model_source = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    model_id = os.path.splitext(os.path.basename(args.model.rstrip('/')))[0]
    print(f'loading {model_source} on {args.backend} ...')
    model = AutoModelForCausalLM.from_pretrained(
        model_source, device_map=args.backend, n_ctx=args.n_ctx
    )
    try:
        if isinstance(model, UniRTVLM):
            capabilities = ', '.join(
                name for name, supported in model.capabilities().items() if supported
            ) or 'no media modality'
            details = f'VLM: {capabilities}'
        else:
            stats = model.runtime_stats()
            details = f"device: {stats['device_name'] or '?'}"
        print(f'ready on http://{args.host}:{args.port}/v1  ({details})')
        serve(model, model_id, args.host, args.port, args.max_queued_requests)
    finally:
        model.close()


if __name__ == '__main__':
    main()

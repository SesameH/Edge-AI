# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI-compatible HTTP server over a UniRT model. Stdlib only.

    python3 -m unirt.server --model models/SmolLM2-135M-Instruct --backend mlx --port 8080

Any of the models may be omitted, so this also serves as a retrieval sidecar:

    python3 -m unirt.server --embedding-model models/all-MiniLM-L6-v2-GGUF \
        --rerank-model models/bge-reranker-v2-m3-GGUF

Endpoints:
    GET  /v1/models
    GET  /v1/stats              (runtime_stats(): memory/device usage; not OpenAI-standard)
    POST /v1/chat/completions   (supports "stream": true via SSE)
    POST /v1/embeddings         (requires --embedding-model)
    POST /v1/rerank             (requires --rerank-model; the Cohere/Jina shape)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import signal
import struct
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .auto import AutoModelForCausalLM, AutoModelForEmbedding
from .modeling import UniRTVLM
from .tool_calling import (
    apply_tool_prompt,
    interpret_output,
    parse_tool_request,
    rewrite_tool_history,
)

_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_MEDIA_BYTES = 12 * 1024 * 1024
# One /v1/embeddings call becomes one native batch, so the whole batch is
# resident at once. This bounds it; clients chunk larger corpora anyway.
_MAX_EMBEDDING_INPUTS = 2048
# Reranking is a cross-encoder pass per document, so a batch costs far more
# than the same number of embeddings.
_MAX_RERANK_DOCUMENTS = 512


def _completion_id() -> str:
    return 'chatcmpl-' + uuid.uuid4().hex[:24]


def _with_serialized_model(model, lock: threading.Lock, operation, *, reuse_prefix=True):
    """Serialize one generation, dropping cached state only when it may be stale.

    Cached KV is what makes multi-turn chat cheap: with n_past=0 the plugin
    reuses the longest prefix shared with the previous transcript, so a resent
    conversation only prefills its new suffix. Resetting on the way out threw
    that away and made every turn re-prefill the whole transcript, which grows
    with the conversation.

    Nothing about the HTTP contract changes -- clients still send the full
    transcript every time, and a prompt that does not extend the cached one
    simply matches a shorter prefix. A failed or abandoned run is the one case
    where the plugin's transcript may no longer mirror its KV, so that path
    still resets.
    """

    with lock:
        try:
            result = operation()
        except BaseException:
            model.reset()
            raise
        if not reuse_prefix:
            model.reset()
        return result


def _parse_embedding_request(req: dict) -> tuple[list[str] | list[list[int]], str]:
    """Validate an OpenAI /v1/embeddings body.

    Returns the inputs as either a list of strings or a list of token rows, and
    the encoding format. `input` accepts all four OpenAI shapes: a string, an
    array of strings, an array of tokens, or an array of token arrays.
    """

    value = req.get('input')
    if isinstance(value, str):
        if not value:
            raise ValueError('input must be a non-empty string or array')
        inputs: list[str] | list[list[int]] = [value]
    elif isinstance(value, list) and value:
        if all(isinstance(item, str) for item in value):
            inputs = list(value)
        elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            inputs = [list(value)]          # one pre-tokenized sequence
        elif all(
            isinstance(row, list)
            and row
            and all(isinstance(item, int) and not isinstance(item, bool) for item in row)
            for row in value
        ):
            inputs = [list(row) for row in value]
        else:
            raise ValueError(
                'input must be a string, an array of strings, an array of tokens, '
                'or an array of token arrays'
            )
    else:
        raise ValueError('input must be a non-empty string or array')

    if len(inputs) > _MAX_EMBEDDING_INPUTS:
        raise ValueError(f'input holds more than {_MAX_EMBEDDING_INPUTS} entries')
    if any(isinstance(item, str) and '\x00' in item for item in inputs):
        raise ValueError('input must not contain NUL bytes')

    encoding_format = req.get('encoding_format', 'float')
    if encoding_format not in ('float', 'base64'):
        raise ValueError("encoding_format must be 'float' or 'base64'")

    # Truncating a vector the model did not train for is silently wrong, and
    # only some models (Matryoshka-trained) survive it, so refuse rather than
    # guess. A client asking for the native width is fine and is checked once
    # the width is known.
    dimensions = req.get('dimensions')
    if dimensions is not None and (
        not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0
    ):
        raise ValueError('dimensions must be a positive integer')

    return inputs, encoding_format


def _parse_rerank_request(req: dict) -> tuple[str, list[str], int | None, bool]:
    """Validate a /v1/rerank body (the Cohere/Jina shape llama.cpp also serves)."""

    query = req.get('query')
    if not isinstance(query, str) or not query or '\x00' in query:
        raise ValueError('query must be a non-empty NUL-free string')

    raw = req.get('documents')
    if not isinstance(raw, list) or not raw:
        raise ValueError('documents must be a non-empty array')
    if len(raw) > _MAX_RERANK_DOCUMENTS:
        raise ValueError(f'documents holds more than {_MAX_RERANK_DOCUMENTS} entries')
    documents: list[str] = []
    for item in raw:
        # Cohere's API accepts objects; its own clients send them.
        text = item.get('text') if isinstance(item, dict) else item
        if not isinstance(text, str) or not text or '\x00' in text:
            raise ValueError(
                'documents must hold non-empty NUL-free strings, or objects with a '
                "non-empty 'text' string"
            )
        documents.append(text)

    top_n = req.get('top_n')
    if top_n is not None and (
        not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0
    ):
        raise ValueError('top_n must be a positive integer')

    return_documents = req.get('return_documents', False)
    if not isinstance(return_documents, bool):
        raise ValueError('return_documents must be a boolean')

    return query, documents, top_n, return_documents


def _relevance(score: float) -> float:
    """Map a cross-encoder logit to (0, 1).

    `relevance_score` means a 0..1 relevance to every client that speaks this
    API, and a raw logit of -11 does not read as one. The sigmoid is what the
    BGE reranker's own model card prescribes, and being monotonic it leaves the
    ranking untouched. Raw logits stay available through UniRTEmbedding.rerank().
    """

    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)       # exp(-score) overflows for very negative scores
    return exponential / (1.0 + exponential)


def _pad_token_rows(
    model, rows: list[list[int]]
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Pad pre-tokenized rows to one rectangle, the way UniRTEmbedding._tokenize does.

    The native encode ABI takes a rectangular batch. Text input gets padded
    inside the model; token input arrives straight from the client, so it has
    to be padded here, on the same side and with the same pad id.
    """

    width = max(len(row) for row in rows)
    pad_id = model._pad_id
    if any(len(row) != width for row in rows) and pad_id is None:
        raise ValueError(
            'token input rows have different lengths and this model has no padding token'
        )
    left = model._padding_side == 'left'
    ids: list[list[int]] = []
    masks: list[list[int]] = []
    for row in rows:
        pad = width - len(row)
        fill = [pad_id or 0] * pad
        ids.append(fill + list(row) if left else list(row) + fill)
        masks.append([0] * pad + [1] * len(row) if left else [1] * len(row) + [0] * pad)
    return ids, masks, [[0] * width for _ in rows]


def _pack_embedding(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == 'float':
        return vector
    # OpenAI's base64 form, which its official Python client requests by
    # default when numpy is installed: little-endian float32, packed.
    return base64.b64encode(struct.pack(f'<{len(vector)}f', *vector)).decode('ascii')


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

    # OpenAI structured outputs: {"type": "json_object"} constrains to
    # syntactically valid JSON; {"type": "json_schema", "json_schema":
    # {"schema": {...}}} constrains decoding to the schema itself.
    response_format = req.get('response_format')
    if response_format is not None:
        if not isinstance(response_format, dict):
            raise ValueError('response_format must be an object')
        kind = response_format.get('type')
        if kind == 'text':
            pass
        elif kind == 'json_object':
            result['json_mode'] = True
        elif kind == 'json_schema':
            wrapper = response_format.get('json_schema')
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get('schema'), dict):
                raise ValueError(
                    "response_format.json_schema must be an object with a 'schema' object"
                )
            result['json_schema'] = wrapper['schema']
        else:
            raise ValueError("response_format.type must be 'text', 'json_object' or 'json_schema'")
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


def _assistant_message(text: str, finish: str, plan) -> tuple[dict, str]:
    """Build the assistant message, splitting out a tool call when tools ran."""
    if plan is None:
        return {'role': 'assistant', 'content': text}, finish
    content, tool_calls = interpret_output(text, plan)
    message: dict = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls
        # 'length' outranks it: a call cut off by max_tokens never parsed, so
        # it cannot reach here, and reporting 'tool_calls' on a truncated turn
        # would tell the client to execute something incomplete.
        if finish != 'length':
            finish = 'tool_calls'
    return message, finish


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

    def _read_json_body(self) -> dict | None:
        """Read and parse one JSON object body, or answer the error and return None."""

        try:
            raw_length = self.headers.get('Content-Length')
            if raw_length is None:
                self._error(411, 'Content-Length is required')
                return None
            length = int(raw_length)
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                self._error(
                    413 if length > _MAX_REQUEST_BYTES else 400, 'invalid request body size'
                )
                return None
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f'bad request body: {exc}')
            return None
        if not isinstance(body, dict):
            self._error(400, 'request body must be a JSON object')
            return None
        return body

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
            listed = [
                {'id': name, 'object': 'model', 'owned_by': 'unirt'}
                for name in (model_id, self.server.embedding_id, self.server.reranker_id)
                if name is not None
            ]
            self._json(200, {'object': 'list', 'data': listed})
        elif self.path == '/v1/stats':
            # Not an OpenAI-standard endpoint: exposes runtime_stats() for
            # UIs/dashboards that want live device/memory info.
            served = next(
                handle
                for handle in (self.server.model, self.server.embedding, self.server.reranker)
                if handle is not None
            )
            self._json(200, served.runtime_stats())
        elif self.path in ('/', '/health'):
            self._json(200, {'status': 'ok', 'model': model_id})
        else:
            self._error(404, f'unknown path {self.path}')

    def do_POST(self):
        self._response_started = False
        if self.path == '/v1/embeddings':
            self._handle_embeddings()
            return
        if self.path == '/v1/rerank':
            self._handle_rerank()
            return
        if self.path != '/v1/chat/completions':
            self._error(404, f'unknown path {self.path}')
            return
        if self.server.model is None:
            self._error(404, 'this server was started without a chat model (--model)')
            return
        req = self._read_json_body()
        if req is None:
            return

        try:
            gen_kwargs = _parse_generation_args(req)
            if not isinstance(req.get('stream', False), bool):
                raise ValueError('stream must be a boolean')
            plan = parse_tool_request(req)
            messages = req.get('messages')
            # Prior calls and results are flattened whether or not this turn
            # declares tools: a client that stops sending `tools` mid-thread
            # still has the earlier tool turns in its transcript.
            if isinstance(messages, list):
                messages = rewrite_tool_history(messages)
            if plan is not None:
                # Both features drive the plugin's single grammar slot.
                if gen_kwargs.get('json_schema') is not None or gen_kwargs.get('json_mode'):
                    raise ValueError(
                        'tools and response_format cannot both constrain one completion'
                    )
                messages = apply_tool_prompt(messages, plan)
                gen_kwargs['json_schema'] = plan.schema()
            prepared = _prepare_messages(
                messages,
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
                    self._stream_completion(prompt, gen_kwargs, plan)
                else:
                    self._blocking_completion(prompt, gen_kwargs, plan)

            _with_serialized_model(
                model,
                self.server.gen_lock,
                generate_response,
                reuse_prefix=self.server.reuse_prefix,
            )
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

    def _handle_embeddings(self) -> None:
        model = self.server.embedding
        if model is None:
            self._error(404, 'this server was started without an embedding model '
                             '(--embedding-model)')
            return
        req = self._read_json_body()
        if req is None:
            return
        try:
            inputs, encoding_format = _parse_embedding_request(req)
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if not self.server.request_slots.acquire(blocking=False):
            self._busy()
            return
        try:
            # Tokenizing here rather than calling encode() is what makes
            # usage.prompt_tokens truthful: encode() tokenizes internally and
            # reports nothing back. UniRTEmbedding serializes itself, so this
            # deliberately does not take gen_lock -- embedding a corpus must not
            # queue behind a long completion.
            if isinstance(inputs[0], str):
                ids, masks, types = model._tokenize(inputs)
            else:
                ids, masks, types = _pad_token_rows(model, inputs)
            vectors = model.encode_tokens(ids, attention_mask=masks, token_type_ids=types)
            tokens = sum(sum(row) for row in masks)

            dimensions = req.get('dimensions')
            if dimensions is not None and vectors and dimensions != len(vectors[0]):
                self._error(
                    400,
                    f'this model emits {len(vectors[0])}-dimensional vectors; '
                    f'dimensions={dimensions} would require truncation, which is '
                    'only valid for Matryoshka-trained models',
                )
                return

            self._json(200, {
                'object': 'list',
                'data': [
                    {
                        'object': 'embedding',
                        'index': index,
                        'embedding': _pack_embedding(vector, encoding_format),
                    }
                    for index, vector in enumerate(vectors)
                ],
                'model': self.server.embedding_id,
                'usage': {'prompt_tokens': tokens, 'total_tokens': tokens},
            })
        except ValueError as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 — HTTP boundary
            if not self._response_started:
                self._error(500, f'embedding failed: {exc}')
            else:
                self.close_connection = True
        finally:
            self.server.request_slots.release()

    def _handle_rerank(self) -> None:
        model = self.server.reranker
        if model is None:
            self._error(404, 'this server was started without a reranker (--rerank-model)')
            return
        req = self._read_json_body()
        if req is None:
            return
        try:
            query, documents, top_n, return_documents = _parse_rerank_request(req)
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if not self.server.request_slots.acquire(blocking=False):
            self._busy()
            return
        try:
            scores = model.rerank(query, documents)
            results = [
                {'index': index, 'relevance_score': _relevance(score)}
                for index, score in enumerate(scores)
            ]
            results.sort(key=lambda entry: entry['relevance_score'], reverse=True)
            if top_n is not None:
                results = results[:top_n]
            if return_documents:
                for entry in results:
                    entry['document'] = {'text': documents[entry['index']]}
            # No usage block: rerank tokenizes inside the plugin, which reports
            # no token counts back, and inventing one would be worse than
            # omitting it.
            self._json(200, {
                'object': 'list',
                'model': self.server.reranker_id,
                'results': results,
            })
        except ValueError as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 — HTTP boundary
            if not self._response_started:
                self._error(500, f'rerank failed: {exc}')
            else:
                self.close_connection = True
        finally:
            self.server.request_slots.release()

    # ---- completion modes ----

    def _blocking_completion(self, prompt: str, gen_kwargs: dict, plan=None) -> None:
        out = self.server.model.generate(prompt, **gen_kwargs)
        p = out.profile
        finish = 'stop' if p.stop_reason in ('eos', 'stop_sequence') else 'length'
        message, finish = _assistant_message(out.text, finish, plan)
        self._json(200, {
            'id': _completion_id(),
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': self.server.model_id,
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': finish,
            }],
            'usage': {
                'prompt_tokens': p.prompt_tokens,
                'completion_tokens': p.generated_tokens,
                'total_tokens': p.prompt_tokens + p.generated_tokens,
            },
        })

    def _stream_completion(self, prompt: str, gen_kwargs: dict, plan=None) -> None:
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
        # With tools the token stream is a half-written JSON envelope, not
        # anything a client can render, so those pieces are accumulated and
        # emitted once as a finished delta. Plain turns still stream live.
        buffered: list[str] = []
        try:
            for piece in streamer:
                if plan is None:
                    send_chunk(delta({'content': piece}))
                else:
                    buffered.append(piece)
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
        if plan is not None:
            message, finish = _assistant_message(''.join(buffered), finish, plan)
            payload = {'content': message['content']}
            if message.get('tool_calls'):
                payload['tool_calls'] = [
                    {'index': index, **call} for index, call in enumerate(message['tool_calls'])
                ]
            send_chunk(delta(payload))
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

    def __init__(
        self,
        address,
        model,
        model_id: str | None,
        max_queued_requests: int = 8,
        embedding=None,
        embedding_id: str | None = None,
        reuse_prefix: bool = True,
        reranker=None,
        reranker_id: str | None = None,
    ):
        self.model = model
        self.model_id = model_id
        self.embedding = embedding
        self.embedding_id = embedding_id
        self.reranker = reranker
        self.reranker_id = reranker_id
        self.reuse_prefix = reuse_prefix
        self.capabilities = model.capabilities() if isinstance(model, UniRTVLM) else None
        self.gen_lock = threading.Lock()
        # Generation itself is fully serialized by gen_lock (one model, one KV
        # state); this only bounds how many callers may be queued waiting for
        # it, so a burst sheds load with 503s instead of piling up unbounded
        # blocked threads.
        self.request_slots = threading.Semaphore(max_queued_requests)
        super().__init__(address, Handler)


def serve(
    model,
    model_id: str | None,
    host: str,
    port: int,
    max_queued_requests: int = 8,
    embedding=None,
    embedding_id: str | None = None,
    reuse_prefix: bool = True,
    reranker=None,
    reranker_id: str | None = None,
) -> None:
    """Serve until interrupted, then stop accepting and join active requests."""

    server = UniRTHTTPServer(
        (host, port), model, model_id, max_queued_requests, embedding, embedding_id,
        reuse_prefix, reranker, reranker_id,
    )
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
        help='local model path or Hugging Face repository id',
    )
    ap.add_argument('--backend', choices=['llama_cpp', 'mlx'], default='llama_cpp')
    ap.add_argument(
        '--embedding-model',
        help='text encoder to serve at /v1/embeddings (local path or Hugging Face '
             'repository id); may be given with or without --model',
    )
    ap.add_argument(
        '--embedding-device',
        default='auto',
        help="device for the embedding model: auto, cpu, gpu, npu (default: auto)",
    )
    ap.add_argument(
        '--rerank-model',
        help='cross-encoder to serve at /v1/rerank (local path or Hugging Face '
             'repository id). A reranker is a different model from an embedding '
             'encoder, so this is a separate flag',
    )
    ap.add_argument(
        '--rerank-device',
        default='auto',
        help='device for the reranker: auto, cpu, gpu, npu (default: auto)',
    )
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument(
        '--n-ctx',
        type=int,
        default=0,
        help='context window in tokens (0 = the model default)',
    )
    ap.add_argument(
        '--no-prefix-cache',
        action='store_true',
        help='clear cached KV state after every request instead of letting the '
             'next one reuse a shared prefix. Slower for multi-turn chat; the '
             'escape hatch for the tiny numeric differences that reusing cached '
             'KV can produce versus a cold prefill',
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
    if not args.model and not args.embedding_model and not args.rerank_model:
        ap.error('at least one of --model, --embedding-model or --rerank-model is required')

    def _source_and_id(path: str) -> tuple[str, str]:
        source = os.path.abspath(path) if os.path.exists(path) else path
        return source, os.path.splitext(os.path.basename(path.rstrip('/')))[0]

    model = None
    model_id = None
    embedding = None
    embedding_id = None
    reranker = None
    reranker_id = None
    served: list[str] = []
    try:
        if args.model:
            model_source, model_id = _source_and_id(args.model)
            print(f'loading {model_source} on {args.backend} ...')
            model = AutoModelForCausalLM.from_pretrained(
                model_source, device_map=args.backend, n_ctx=args.n_ctx
            )
            if isinstance(model, UniRTVLM):
                capabilities = ', '.join(
                    name for name, supported in model.capabilities().items() if supported
                ) or 'no media modality'
                served.append(f'VLM: {capabilities}')
            else:
                stats = model.runtime_stats()
                served.append(f"chat on {stats['device_name'] or '?'}")

        if args.embedding_model:
            embedding_source, embedding_id = _source_and_id(args.embedding_model)
            print(f'loading {embedding_source} for embeddings ...')
            embedding = AutoModelForEmbedding.from_pretrained(
                embedding_source, device_map=args.embedding_device
            )
            stats = embedding.runtime_stats()
            served.append(f"embeddings on {stats.get('device_name') or '?'}")

        if args.rerank_model:
            rerank_source, reranker_id = _source_and_id(args.rerank_model)
            print(f'loading {rerank_source} for reranking ...')
            reranker = AutoModelForEmbedding.from_pretrained(
                rerank_source, device_map=args.rerank_device
            )
            stats = reranker.runtime_stats()
            served.append(f"rerank on {stats.get('device_name') or '?'}")

        print(f"ready on http://{args.host}:{args.port}/v1  ({'; '.join(served)})")
        serve(
            model,
            model_id,
            args.host,
            args.port,
            args.max_queued_requests,
            embedding,
            embedding_id,
            not args.no_prefix_cache,
            reranker,
            reranker_id,
        )
    finally:
        for handle in (model, embedding, reranker):
            if handle is not None:
                handle.close()


if __name__ == '__main__':
    main()

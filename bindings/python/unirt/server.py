# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI-compatible HTTP server over a UniRT model. Stdlib only.

    python3 -m unirt.server --model models/SmolLM2-135M-Instruct --backend mlx --port 8080

--model may be repeated, and the "model" field of a request then picks between
them; all but the first load on demand:

    python3 -m unirt.server --model small=models/SmolLM2-135M-Instruct-Q8_0.gguf \
        --model gemma=models/gemma-3-270m-it-Q8_0.gguf --max-resident-models 1

Any of the models may be omitted, so this also serves as a retrieval sidecar:

    python3 -m unirt.server --embedding-model models/all-MiniLM-L6-v2-GGUF \
        --rerank-model models/bge-reranker-v2-m3-GGUF

Endpoints:
    GET  /v1/models
    GET  /v1/stats              (runtime_stats(): memory/device usage; not OpenAI-standard)
    POST /v1/chat/completions   (supports "stream": true via SSE)
    POST /v1/completions        (the pre-chat shape: prompt in, no chat template)
    POST /v1/embeddings         (requires --embedding-model)
    POST /v1/rerank             (requires --rerank-model; the Cohere/Jina shape)

--api-key (or UNIRT_API_KEY) requires Authorization: Bearer <key> on every /v1
endpoint; GET / and /health stay open for liveness probes.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import binascii
import hmac
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
# /v1/completions takes an array of prompts, and each one is a full generation
# run held on a single serialized model.
_MAX_COMPLETION_PROMPTS = 16
# Alternatives reported per token. Each one is a token piece plus its bytes in
# the response, so a large value inflates the reply far more than the request;
# OpenAI caps top_logprobs at 20 and there is no reason to be looser.
_MAX_TOP_LOGPROBS = 20


def _completion_id(prefix: str = 'chatcmpl') -> str:
    return f'{prefix}-' + uuid.uuid4().hex[:24]


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
        # A policy the server picks, not something the plugins do on their own:
        # a chat that outgrows the context should keep answering rather than
        # start returning errors, and dropping the oldest turns is the least
        # bad way to do that. The library refuses to make that choice silently
        # -- see sliding_window in unirt.h -- which is why it is spelled out
        # here where it can be turned off.
        'sliding_window': True,
    }

    # presence_penalty and frequency_penalty are standard OpenAI fields and
    # top_k / min_p / repetition_penalty are what every local runtime exposes;
    # the sampler behind unirt_SamplerConfig takes all five. Passing them
    # through beats accepting them and quietly changing nothing, which is
    # indistinguishable from the parameter having no effect on the model.
    for name, low, high in (
        ('top_k', 0, 2**31 - 1),
        ('min_p', 0.0, 1.0),
        ('repetition_penalty', 0.0, 4.0),
        ('presence_penalty', -2.0, 2.0),
        ('frequency_penalty', -2.0, 2.0),
    ):
        value = req.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number')
        if name == 'top_k':
            if not isinstance(value, int):
                raise ValueError('top_k must be an integer')
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        result[name] = value

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


def _conversation_key(messages) -> str:
    """A string whose prefixes grow with the conversation, for slot affinity.

    Only ever compared prefix-wise against another one of these, so the exact
    encoding does not matter -- only that appending a turn appends to the key
    rather than rewriting it. Multimodal parts collapse to their type: a slot
    holding this conversation's text prefix is still the right one.
    """

    parts = []
    for message in messages:
        content = message.get('content', '')
        if not isinstance(content, str):
            content = ' '.join(
                block.get('text', block.get('type', ''))
                for block in content
                if isinstance(block, dict)
            )
        parts.append(f"{message.get('role', '')}\x1f{content}")
    return '\x1e'.join(parts)


class _Slot:
    """One decoding slot: a model handle and the KV cache that belongs to it."""

    __slots__ = ('model', 'affinity')

    def __init__(self, model) -> None:
        self.model = model
        # What this slot's cached KV was last built from. Not the rendered
        # prompt: rendering has to happen on the slot's own handle, and that
        # handle may be mid-generation when the choice is being made.
        self.affinity = ''


class SlotPool:
    """Several handles on one model, each decoding independently.

    The server used to hold one handle behind one lock, so a second request
    waited for the first to finish generating -- fine for a desktop chat
    window, useless for anything else. A slot is a separate llama_context with
    its own KV cache; the weights behind them are shared inside the plugin, so
    N slots cost N key-value caches rather than N copies of the model.

    Which slot a request gets is not arbitrary. Each one remembers what its
    cache was last built from, and an incoming request goes to the free slot
    with the longest shared prefix, so a conversation returns to the slot that
    already holds it and prefills only its new turn. Round-robin would throw
    that away on every request.
    """

    def __init__(self, models: list) -> None:
        if not models:
            raise ValueError('a slot pool needs at least one model handle')
        self._slots = [_Slot(model) for model in models]
        self._free = list(self._slots)
        self._condition = threading.Condition()

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def models(self) -> list:
        return [slot.model for slot in self._slots]

    @staticmethod
    def _score(slot: _Slot, affinity: str) -> int:
        """How much this slot's cache is worth to this request.

        Continuation, not similarity. A next turn *extends* the key its own
        previous turn stored, so the test is startswith, not longest common
        prefix -- two unrelated conversations still share the leading role
        marker, and scoring that made a new conversation evict a live one in
        preference to taking an idle slot.
        """

        if slot.affinity and affinity.startswith(slot.affinity):
            return len(slot.affinity)
        # An idle slot costs nothing to take; a stranger's slot throws away a
        # cache that is still worth something to whoever built it.
        return 0 if not slot.affinity else -1

    def _take(self, affinity: str) -> _Slot:
        best = max(
            range(len(self._free)),
            key=lambda index: self._score(self._free[index], affinity),
        )
        return self._free.pop(best)

    @contextlib.contextmanager
    def checkout(self, affinity: str, *, reuse_prefix: bool = True, timeout: float | None = None):
        """Lease a slot, or raise TimeoutError if none frees up in time."""
        with self._condition:
            if not self._free and not self._condition.wait_for(
                lambda: bool(self._free), timeout=timeout
            ):
                raise TimeoutError('no decoding slot became free')
            slot = self._take(affinity)
        try:
            yield slot
        except BaseException:
            # A run that raised or was abandoned may leave the plugin's
            # transcript out of step with its KV, and the next request on this
            # slot would then reuse a prefix that is not really there.
            slot.model.reset()
            slot.affinity = ''
            raise
        else:
            if reuse_prefix:
                slot.affinity = affinity
            else:
                slot.model.reset()
                slot.affinity = ''
        finally:
            with self._condition:
                self._free.append(slot)
                self._condition.notify()

    def close(self) -> None:
        for slot in self._slots:
            slot.model.close()


@dataclass(frozen=True)
class ModelSpec:
    """How to load one named model, and how many slots to give it."""

    name: str
    source: str
    backend: str = 'llama_cpp'
    n_ctx: int = 0
    slots: int = 1
    draft: str | None = None


class _Resident:
    """A loaded model: its pool of slots, and what is keeping it loaded."""

    __slots__ = ('spec', 'pool', 'model', 'capabilities', 'users', 'idle_since')

    def __init__(self, spec: ModelSpec, pool: SlotPool) -> None:
        self.spec = spec
        self.pool = pool
        # One handle stands for all of them when the question is about the
        # model rather than about a decoding slot: modality, capabilities,
        # runtime stats.
        self.model = pool.models[0]
        self.capabilities = (
            self.model.capabilities() if isinstance(self.model, UniRTVLM) else None
        )
        self.users = 0
        self.idle_since = time.monotonic()

    @property
    def name(self) -> str:
        return self.spec.name


def load_slots(spec: ModelSpec) -> list:
    """Open a model's handles: one per decoding slot, over shared weights."""
    def open_one():
        return AutoModelForCausalLM.from_pretrained(
            spec.source, device_map=spec.backend, n_ctx=spec.n_ctx,
            **({'draft_model': spec.draft} if spec.draft else {}),
        )

    first = open_one()
    if isinstance(first, UniRTVLM):
        # Media position state is per handle, so a VLM gets exactly one slot
        # whatever was asked for.
        if spec.slots > 1:
            print(f'note: {spec.name} is a VLM and gets one slot, not {spec.slots}')
        return [first]
    return [first] + [open_one() for _ in range(spec.slots - 1)]


class ModelRegistry:
    """The chat models this server can serve, by name.

    A model loads when a request first names it, and stays loaded while
    anything is using it. That is what makes several models affordable on a
    device that could not hold them all at once: configuring a model costs
    nothing until it is asked for, and `--max-resident-models` /
    `--model-idle-timeout` decide when one is given back.

    The `model` field of a request picks one. With a single model configured
    the field is not checked -- there is then no other model the request could
    wrongly be served by, and clients that hardcode a name would break for no
    gain. With several, an unknown name is a 404 rather than a silent answer
    from whichever model happened to be loaded.
    """

    def __init__(
        self,
        specs: list[ModelSpec],
        *,
        loader=load_slots,
        resident_limit: int | None = None,
        idle_timeout: float = 0.0,
    ) -> None:
        if not specs:
            raise ValueError('a model registry needs at least one model')
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError('two models cannot share a name')
        self._loader = loader
        self._resident_limit = resident_limit or len(specs)
        self._idle_timeout = idle_timeout
        self._resident: dict[str, _Resident] = {}
        self._lock = threading.Lock()
        # Per model, so that loading one does not hold up requests for another.
        self._load_locks = {name: threading.Lock() for name in self._specs}
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None
        self.default_name = specs[0].name
        if idle_timeout > 0:
            self._reaper = threading.Thread(
                target=self._reap_idle, name='unirt-model-reaper', daemon=True
            )
            self._reaper.start()

    def names(self) -> list[str]:
        return list(self._specs)

    @property
    def routed(self) -> bool:
        """Whether the `model` field of a request has anything to choose."""
        return len(self._specs) > 1

    @property
    def total_slots(self) -> int:
        return sum(spec.slots for spec in self._specs.values())

    def resolve(self, requested) -> str:
        """The name to serve, or KeyError if the client asked for a stranger."""
        if requested is None or not self.routed:
            return self.default_name
        if not isinstance(requested, str):
            raise ValueError('model must be a string')
        if requested not in self._specs:
            raise KeyError(requested)
        return requested

    def loaded(self) -> list[_Resident]:
        with self._lock:
            return list(self._resident.values())

    @contextlib.contextmanager
    def acquire(self, name: str):
        """Pin a model as loaded for the duration of the block."""
        entry = self._pin(name)
        try:
            yield entry
        finally:
            with self._lock:
                entry.users -= 1
                if entry.users <= 0:
                    entry.idle_since = time.monotonic()
            # Releasing is the moment something becomes evictable. Without
            # this, a registry pushed over its limit because everything was
            # in use would stay over it until the next load.
            self._evict_over_limit()

    def preload(self, name: str) -> _Resident:
        """Load a model now rather than on first use. Startup uses this for
        the default model, so a bad path or a missing plugin is reported while
        someone is still watching the terminal."""
        with self.acquire(name) as entry:
            return entry

    def _pin(self, name: str) -> _Resident:
        with self._lock:
            entry = self._resident.get(name)
            if entry is not None:
                entry.users += 1
                return entry
            spec = self._specs[name]

        # Loading takes seconds and must happen outside the registry lock, or
        # a request for an already-resident model would queue behind it. The
        # per-model lock is what keeps two callers from loading it twice.
        with self._load_locks[name]:
            with self._lock:
                entry = self._resident.get(name)
                if entry is not None:
                    entry.users += 1
                    return entry
            pool = SlotPool(self._loader(spec))
            entry = _Resident(spec, pool)
            with self._lock:
                entry.users += 1
                self._resident[name] = entry
        self._evict_over_limit()
        return entry

    def _evict_over_limit(self) -> None:
        while True:
            with self._lock:
                if len(self._resident) <= self._resident_limit:
                    return
                idle = [entry for entry in self._resident.values() if entry.users <= 0]
                if not idle:
                    # Everything resident is in use. Staying over the limit
                    # beats closing a model out from under a live request; the
                    # next release brings it back down.
                    return
                victim = min(idle, key=lambda entry: entry.idle_since)
                del self._resident[victim.name]
            victim.pool.close()

    def _reap_idle(self) -> None:
        # Wake often enough to be roughly punctual without spinning on a long
        # timeout.
        interval = max(1.0, min(self._idle_timeout, 30.0))
        while not self._stop.wait(interval):
            deadline = time.monotonic() - self._idle_timeout
            with self._lock:
                stale = [
                    entry
                    for entry in self._resident.values()
                    if entry.users <= 0 and entry.idle_since <= deadline
                ]
                for entry in stale:
                    del self._resident[entry.name]
            for entry in stale:
                entry.pool.close()

    def close(self) -> None:
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
        with self._lock:
            entries = list(self._resident.values())
            self._resident.clear()
        for entry in entries:
            entry.pool.close()


def _registry_of(handles: list, name: str) -> ModelRegistry:
    """A registry over handles that are already open.

    The single `--model` path and the tests hand the server live handles
    rather than something to load, so the one model is resident from the
    start and there is nothing to reload it from.
    """

    registry = ModelRegistry(
        [ModelSpec(name=name, source='', slots=len(handles))],
        loader=lambda _spec: handles,
    )
    registry.preload(name)
    return registry


def _parse_logprobs_request(req: dict, *, chat: bool) -> int:
    """How many alternatives to report per token, in the shape each endpoint uses.

    Chat spells it `logprobs: true` plus an optional `top_logprobs: N`; the
    older completions endpoint spells it `logprobs: N` directly. Both land on
    the same native request, so they are parsed into the same number.
    """

    if chat:
        wanted = req.get('logprobs', False)
        if not isinstance(wanted, bool):
            raise ValueError('logprobs must be a boolean on /v1/chat/completions')
        top = req.get('top_logprobs')
        if top is None:
            return 1 if wanted else 0
        if not wanted:
            raise ValueError('top_logprobs requires logprobs to be true')
        if not isinstance(top, int) or isinstance(top, bool) or not 0 <= top <= _MAX_TOP_LOGPROBS:
            raise ValueError(f'top_logprobs must be an integer between 0 and {_MAX_TOP_LOGPROBS}')
        # The native side always reports the sampled token; top_logprobs counts
        # the alternatives beside it, and 0 of those still means "report it".
        return max(top, 1)

    wanted = req.get('logprobs')
    if wanted is None:
        return 0
    if not isinstance(wanted, int) or isinstance(wanted, bool) or not 0 <= wanted <= _MAX_TOP_LOGPROBS:
        raise ValueError(f'logprobs must be an integer between 0 and {_MAX_TOP_LOGPROBS}')
    return max(wanted, 1)


def _logprob_entry(value) -> dict:
    return {
        'token': value.token,
        'logprob': value.logprob,
        # OpenAI sends the raw bytes so a client can rebuild a character that
        # one token only carries part of; `token` alone cannot express that.
        'bytes': list(value.token.encode('utf-8')),
    }


def _pending_steps(streamer, already_reported: int):
    """The logprob steps behind the chunk just yielded, or None if unasked.

    A chunk is not a token: the bridge holds a piece back until the bytes that
    finish its character arrive, so one chunk can cover several tokens. Slicing
    by how many entries existed at the previous chunk is what keeps every token
    attached to the text it actually produced.
    """

    steps = getattr(streamer, 'logprobs', None)
    if steps is None:
        return None
    return steps[already_reported:]


def _pending_logprobs(streamer, already_reported: int):
    """Chat-shaped logprobs for the chunk just yielded."""
    steps = _pending_steps(streamer, already_reported)
    return _logprob_payload(steps) if steps else None


def _logprob_payload(steps) -> dict:
    """The chat endpoint's `logprobs` object: one entry per token, in order."""
    return {
        'content': [
            {
                **_logprob_entry(step.chosen),
                'top_logprobs': [_logprob_entry(alternative) for alternative in step.top],
            }
            for step in steps
        ]
    }


def _legacy_logprob_payload(steps, base_offset: int) -> dict:
    """The `logprobs` object /v1/completions uses, which is not the chat one.

    The older endpoint predates the per-token object list: it carries four
    parallel arrays instead, and `top_logprobs` is a token->logprob mapping per
    position rather than a list. Clients built on the official schema reject
    the chat shape here outright, so the two cannot be shared.
    """

    tokens = [step.chosen.token for step in steps]
    offsets = []
    cursor = base_offset
    for token in tokens:
        offsets.append(cursor)
        cursor += len(token)
    return {
        'tokens': tokens,
        'token_logprobs': [step.chosen.logprob for step in steps],
        # A mapping loses duplicate token strings within one position; that is
        # the format's own limitation, not something to work around here.
        'top_logprobs': [
            {alternative.token: alternative.logprob for alternative in step.top}
            for step in steps
        ],
        'text_offset': offsets,
    }


def _parse_completion_request(req: dict) -> tuple[list[str], bool]:
    """Validate a legacy /v1/completions body. Returns (prompts, echo).

    The prompt goes to the model verbatim -- no chat template. That is the
    whole point of this endpoint next to /v1/chat/completions, and the reason
    it is still worth having: base models, fill-in-style prompting, and the
    older clients that only ever learned this shape.
    """

    value = req.get('prompt')
    if isinstance(value, str):
        prompts = [value]
    elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        prompts = list(value)
    elif isinstance(value, list) and value:
        # OpenAI also accepts pre-tokenized prompts here. generate() takes text,
        # and silently detokenizing someone's ids would be a different prompt.
        raise ValueError(
            'prompt must be a string or an array of strings; token-array prompts '
            'are not supported by this server'
        )
    else:
        raise ValueError('prompt must be a non-empty string or array of strings')
    if any(not prompt for prompt in prompts):
        # OpenAI defaults an absent prompt to <|endoftext|>; guessing a token
        # for a local model whose vocabulary may not have it would be worse
        # than saying so.
        raise ValueError('prompt entries must be non-empty')
    if any('\x00' in prompt for prompt in prompts):
        raise ValueError('prompt must not contain NUL bytes')
    if len(prompts) > _MAX_COMPLETION_PROMPTS:
        raise ValueError(f'prompt holds more than {_MAX_COMPLETION_PROMPTS} entries')

    # Each of these would change what the caller gets back, so accepting and
    # ignoring them is worse than refusing: the reply would look right.
    for name in ('n', 'best_of'):
        count = req.get(name)
        if count is not None and count != 1:
            raise ValueError(f'{name} must be 1; this server generates one completion per prompt')
    if req.get('suffix') is not None:
        raise ValueError('suffix (fill-in-the-middle) is not supported')
    echo = req.get('echo', False)
    if not isinstance(echo, bool):
        raise ValueError('echo must be a boolean')
    return prompts, echo


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
        # Every decoding slot is busy and the queue behind them is full, or a
        # request waited out --slot-timeout. Shedding with a clear signal beats
        # piling up blocked threads a client cannot see.
        self._json(
            503,
            {'error': {
                'message': 'server is busy handling other requests; retry shortly',
                'type': 'server_error',
            }},
            extra_headers={'Retry-After': '1'},
        )

    def _authorized(self) -> bool:
        """Check the bearer token when one is configured.

        Without --api-key the server is open, which is right for the default
        127.0.0.1 bind and wrong the moment anyone passes --host 0.0.0.0.
        Compared with compare_digest so the check does not leak the key's
        length or a matching prefix through timing.
        """

        expected = self.server.api_key
        if expected is None:
            return True
        header = self.headers.get('Authorization', '')
        scheme, _, token = header.partition(' ')
        # compare_digest refuses str arguments holding non-ASCII, so a key or a
        # presented token with any such character raises TypeError instead of
        # returning False -- an unauthenticated caller could take the request
        # thread down that path at will. Comparing the UTF-8 bytes is defined
        # for every input and still constant-time.
        if scheme.lower() != 'bearer' or not hmac.compare_digest(
            token.strip().encode('utf-8'), expected.encode('utf-8')
        ):
            self._json(
                401,
                {'error': {
                    'message': 'missing or invalid API key; send '
                               'Authorization: Bearer <key>',
                    'type': 'invalid_request_error',
                    'code': 'invalid_api_key',
                }},
                extra_headers={'WWW-Authenticate': 'Bearer'},
            )
            return False
        return True

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

    def _resolve_model(self, req: dict) -> str | None:
        """Which model this request named, or None once the error is answered.

        With one model configured the field is not checked: there is nothing
        else it could be routed to, and clients that hardcode an OpenAI model
        name would break for no gain. With several, answering from the wrong
        one is exactly the failure this endpoint has to avoid.
        """

        registry = self.server.registry
        try:
            return registry.resolve(req.get('model'))
        except ValueError as exc:
            self._error(400, str(exc))
        except KeyError:
            self._error(
                404,
                f"the model '{req.get('model')}' is not served here; "
                f"this server has {', '.join(registry.names())}",
            )
        return None

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
        if self.path in ('/', '/health'):
            # Deliberately unauthenticated: liveness probes and container
            # healthchecks should not need the key, and it discloses nothing
            # beyond the fact that a server is listening.
            self._json(200, {'status': 'ok', 'model': model_id})
            return
        if not self._authorized():
            return
        if self.path == '/v1/models':
            registry = self.server.registry
            chat_models = registry.names() if registry is not None else []
            listed = [
                {'id': name, 'object': 'model', 'owned_by': 'unirt'}
                for name in (
                    *chat_models, self.server.embedding_id, self.server.reranker_id
                )
                if name is not None
            ]
            self._json(200, {'object': 'list', 'data': listed})
        elif self.path == '/v1/stats':
            # Not an OpenAI-standard endpoint: exposes runtime_stats() for
            # UIs/dashboards that want live device/memory info.
            registry = self.server.registry
            # Whichever chat model is resident right now -- with several
            # configured, only the ones that have been asked for are loaded,
            # and an unloaded model has no runtime to report.
            resident = registry.loaded() if registry is not None else []
            resident.sort(key=lambda entry: entry.name != model_id)
            served = next(
                (
                    handle
                    for handle in (
                        resident[0].model if resident else None,
                        self.server.embedding,
                        self.server.reranker,
                    )
                    if handle is not None
                ),
                None,
            )
            if served is None and registry is None:
                self._error(404, 'this server was started with no model to report on')
                return
            # A configured model that has been unloaded (idle timeout, or
            # evicted) has no runtime to report, but the server is still
            # serving it -- 404 would read as "no such endpoint" to a
            # dashboard polling this.
            stats = served.runtime_stats() if served is not None else {}
            if registry is not None:
                stats['loaded_models'] = [entry.name for entry in resident]
            self._json(200, stats)
        else:
            self._error(404, f'unknown path {self.path}')

    def do_POST(self):
        self._response_started = False
        if not self._authorized():
            return
        if self.path == '/v1/embeddings':
            self._handle_embeddings()
            return
        if self.path == '/v1/rerank':
            self._handle_rerank()
            return
        if self.path == '/v1/completions':
            self._handle_completions()
            return
        if self.path != '/v1/chat/completions':
            self._error(404, f'unknown path {self.path}')
            return
        if self.server.registry is None:
            self._error(404, 'this server was started without a chat model (--model)')
            return
        req = self._read_json_body()
        if req is None:
            return
        served_name = self._resolve_model(req)
        if served_name is None:
            return

        try:
            gen_kwargs = _parse_generation_args(req)
            if not isinstance(req.get('stream', False), bool):
                raise ValueError('stream must be a boolean')
            gen_kwargs['logprobs'] = _parse_logprobs_request(req, chat=True)
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
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if not self.server.request_slots.acquire(blocking=False):
            self._busy()
            return

        prepared = None
        try:
            # Loading happens here rather than at startup for every model but
            # the default, so a request pays for the model it named and for no
            # other.
            with self.server.registry.acquire(served_name) as served:
                try:
                    # Whether media is allowed is the served model's answer,
                    # so it cannot be settled before the model is known.
                    prepared = _prepare_messages(
                        messages,
                        multimodal=isinstance(served.model, UniRTVLM),
                        capabilities=served.capabilities,
                    )
                except ValueError as exc:
                    self._error(400, str(exc))
                    return
                # The affinity key is the conversation, not the rendered
                # prompt: rendering runs on the slot's own handle, which may
                # still be generating while the choice is being made.
                # Rendering is a pure function of the messages, so a growing
                # conversation shares a prefix here exactly as it will there.
                with served.pool.checkout(
                    _conversation_key(prepared.messages),
                    reuse_prefix=self.server.reuse_prefix,
                    timeout=self.server.slot_timeout,
                ) as slot:
                    model = slot.model
                    prompt = model._apply_chat_template(prepared.messages, True, False, None)
                    if isinstance(model, UniRTVLM):
                        gen_kwargs['images'] = prepared.images
                        gen_kwargs['audios'] = prepared.audios
                    if req.get('stream'):
                        self._stream_completion(model, prompt, gen_kwargs, served_name, plan)
                    else:
                        self._blocking_completion(model, prompt, gen_kwargs, served_name, plan)
        except TimeoutError:
            self._busy()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 — HTTP boundary
            if not self._response_started:
                self._error(500, f'generation failed: {exc}')
            else:
                self.close_connection = True
        finally:
            self.server.request_slots.release()
            if prepared is not None:
                prepared.cleanup()

    def _handle_completions(self) -> None:
        """POST /v1/completions -- the pre-chat endpoint, prompt in, text out.

        No chat template is applied: the prompt reaches the model exactly as
        sent. Clients that only speak this shape, and base models with no chat
        template at all, both need that.
        """

        if self.server.registry is None:
            self._error(404, 'this server was started without a chat model (--model)')
            return
        req = self._read_json_body()
        if req is None:
            return
        served_name = self._resolve_model(req)
        if served_name is None:
            return
        try:
            gen_kwargs = _parse_generation_args(req)
            stream = req.get('stream', False)
            if not isinstance(stream, bool):
                raise ValueError('stream must be a boolean')
            prompts, echo = _parse_completion_request(req)
            gen_kwargs['logprobs'] = _parse_logprobs_request(req, chat=False)
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if not self.server.request_slots.acquire(blocking=False):
            self._busy()
            return
        try:
            with self.server.registry.acquire(served_name) as served:
                if isinstance(served.model, UniRTVLM):
                    self._error(
                        400, f"/v1/completions is text-only; '{served_name}' is a VLM"
                    )
                    return
                with served.pool.checkout(
                    # The prompts run one after another on the one slot, so
                    # what its cache holds afterwards is the last of them --
                    # that is the affinity the next request should match.
                    prompts[-1],
                    reuse_prefix=self.server.reuse_prefix,
                    timeout=self.server.slot_timeout,
                ) as slot:
                    if stream:
                        self._stream_text_completions(
                            slot.model, prompts, gen_kwargs, echo, served_name)
                    else:
                        self._blocking_text_completions(
                            slot.model, prompts, gen_kwargs, echo, served_name)
        except TimeoutError:
            self._busy()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 — HTTP boundary
            if not self._response_started:
                self._error(500, f'generation failed: {exc}')
            else:
                self.close_connection = True
        finally:
            self.server.request_slots.release()

    def _blocking_text_completions(
        self, model, prompts: list[str], gen_kwargs: dict, echo: bool, model_id: str
    ) -> None:
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, prompt in enumerate(prompts):
            out = model.generate(prompt, **gen_kwargs)
            profile = out.profile
            prompt_tokens += profile.prompt_tokens
            completion_tokens += profile.generated_tokens
            choices.append({
                'index': index,
                'text': (prompt + out.text) if echo else out.text,
                # text_offset counts from the start of this choice's text, so
                # echoing the prompt shifts every token along by its length.
                'logprobs': (
                    _legacy_logprob_payload(out.logprobs, len(prompt) if echo else 0)
                    if out.logprobs else None
                ),
                'finish_reason': (
                    'stop' if profile.stop_reason in ('eos', 'stop_sequence') else 'length'
                ),
            })
        self._json(200, {
            'id': _completion_id('cmpl'),
            'object': 'text_completion',
            'created': int(time.time()),
            'model': model_id,
            'choices': choices,
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            },
        })

    def _stream_text_completions(
        self, model, prompts: list[str], gen_kwargs: dict, echo: bool, model_id: str
    ) -> None:
        self._begin_event_stream()
        cid = _completion_id('cmpl')
        created = int(time.time())

        def chunk(index: int, text: str, finish=None, logprobs=None) -> dict:
            return {
                'id': cid, 'object': 'text_completion', 'created': created,
                'model': model_id,
                'choices': [{
                    'index': index, 'text': text, 'logprobs': logprobs,
                    'finish_reason': finish,
                }],
            }

        # Prompts run one after another rather than interleaved: this request
        # holds one slot, which is one KV cache, so there is no concurrency to
        # express within it.
        for index, prompt in enumerate(prompts):
            if echo:
                self._send_event(chunk(index, prompt))
            streamer = model.generate(prompt, stream=True, **gen_kwargs)
            reported = 0
            offset = len(prompt) if echo else 0
            try:
                for piece in streamer:
                    steps = _pending_steps(streamer, reported)
                    self._send_event(chunk(
                        index, piece,
                        logprobs=(
                            _legacy_logprob_payload(steps, offset) if steps else None
                        ),
                    ))
                    if steps is not None:
                        reported += len(steps)
                    offset += len(piece)
            except (BrokenPipeError, ConnectionResetError):
                self._drain_cancelled(streamer)
                raise
            out = streamer.output
            finish = (
                'stop' if out and out.profile.stop_reason in ('eos', 'stop_sequence')
                else 'length'
            )
            # Same as the chat stream: the steps behind the last piece of text
            # (a trimmed stop-sequence token) still belong in the reply.
            trailing = _pending_steps(streamer, reported)
            self._send_event(chunk(
                index, '', finish=finish,
                logprobs=_legacy_logprob_payload(trailing, offset) if trailing else None,
            ))
        self._end_event_stream()

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
            # deliberately does not take a decoding slot -- embedding a corpus
            # must not queue behind a long completion.
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

    def _blocking_completion(
        self, model, prompt: str, gen_kwargs: dict, model_id: str, plan=None
    ) -> None:
        out = model.generate(prompt, **gen_kwargs)
        p = out.profile
        finish = 'stop' if p.stop_reason in ('eos', 'stop_sequence') else 'length'
        message, finish = _assistant_message(out.text, finish, plan)
        self._json(200, {
            'id': _completion_id(),
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model_id,
            'choices': [{
                'index': 0,
                'message': message,
                'logprobs': _logprob_payload(out.logprobs) if out.logprobs else None,
                'finish_reason': finish,
            }],
            'usage': {
                'prompt_tokens': p.prompt_tokens,
                'completion_tokens': p.generated_tokens,
                'total_tokens': p.prompt_tokens + p.generated_tokens,
            },
        })

    def _begin_event_stream(self) -> None:
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        self._response_started = True

    def _send_event(self, payload: dict) -> None:
        data = b'data: ' + json.dumps(payload).encode() + b'\n\n'
        self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
        self.wfile.flush()

    def _end_event_stream(self) -> None:
        done = b'data: [DONE]\n\n'
        self.wfile.write(f'{len(done):x}\r\n'.encode() + done + b'\r\n')
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()

    def _drain_cancelled(self, streamer) -> None:
        """Stop a generation whose client has gone, without leaving the model mid-run.

        The plugin keeps a token transcript that has to mirror its KV cache, so
        the run must be allowed to unwind rather than abandoned; the caller then
        resets anyway, but only after this returns.
        """
        streamer.cancel()
        try:
            for _ in streamer:
                pass
        except BaseException:
            pass

    def _stream_completion(
        self, model, prompt: str, gen_kwargs: dict, model_id: str, plan=None
    ) -> None:
        self._begin_event_stream()

        cid = _completion_id()
        created = int(time.time())
        send_chunk = self._send_event

        def delta(d: dict, finish=None, logprobs=None) -> dict:
            choice: dict = {'index': 0, 'delta': d, 'finish_reason': finish}
            if logprobs is not None:
                choice['logprobs'] = logprobs
            return {
                'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                'model': model_id,
                'choices': [choice],
            }

        send_chunk(delta({'role': 'assistant', 'content': ''}))
        streamer = model.generate(prompt, stream=True, **gen_kwargs)
        # With tools the token stream is a half-written JSON envelope, not
        # anything a client can render, so those pieces are accumulated and
        # emitted once as a finished delta. Plain turns still stream live.
        buffered: list[str] = []
        reported = 0
        try:
            for piece in streamer:
                if plan is None:
                    send_chunk(delta(
                        {'content': piece},
                        logprobs=_pending_logprobs(streamer, reported),
                    ))
                else:
                    buffered.append(piece)
                if streamer.logprobs is not None:
                    reported = len(streamer.logprobs)
        except (BrokenPipeError, ConnectionResetError):
            self._drain_cancelled(streamer)
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
        # Tokens can be decoded after the last piece of text: a stop sequence
        # is recognised from a token whose text is then trimmed away. Blocking
        # replies report those steps, so the stream has to as well, or the same
        # request answers with a different number of logprobs depending only on
        # how it was asked for.
        send_chunk(delta({}, finish=finish, logprobs=_pending_logprobs(streamer, reported)))
        self._end_event_stream()


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
        api_key: str | None = None,
        slot_timeout: float = 120.0,
        registry: ModelRegistry | None = None,
    ):
        # Either a registry (several named models, loaded on demand) or the
        # handles for one, which is the same thing with one entry in it.
        # `model` may be one handle or several on the same weights; several is
        # what lets requests decode at the same time.
        if registry is None:
            handles = (
                list(model) if isinstance(model, (list, tuple)) else ([model] if model else [])
            )
            registry = (
                _registry_of(handles, model_id or 'model') if handles else None
            )
        self.registry = registry
        self.model_id = model_id if registry is None else registry.default_name
        self.embedding = embedding
        self.embedding_id = embedding_id
        self.reranker = reranker
        self.reranker_id = reranker_id
        self.api_key = api_key
        self.reuse_prefix = reuse_prefix
        # Bounds how many callers may be in the server at once, so a burst
        # sheds load with 503s instead of piling up unbounded blocked threads.
        # It has to leave room for the running requests as well as the queued
        # ones: sized at the queue depth alone, a pool with more slots than
        # that would have had slots it could never hand out.
        configured_slots = registry.total_slots if registry is not None else 0
        self.request_slots = threading.Semaphore(configured_slots + max_queued_requests)
        # How long a request waits for a slot before giving up with a 503.
        # Long enough to ride out a normal turn ahead of it, short enough that
        # a client is not left hanging behind a queue it cannot see.
        self.slot_timeout = slot_timeout
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
    api_key: str | None = None,
    slot_timeout: float = 120.0,
    registry: ModelRegistry | None = None,
) -> None:
    """Serve until interrupted, then stop accepting and join active requests."""

    server = UniRTHTTPServer(
        (host, port), model, model_id, max_queued_requests, embedding, embedding_id,
        reuse_prefix, reranker, reranker_id, api_key, slot_timeout, registry,
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
        # The registry owns the chat handles -- with lazy loading, the caller
        # cannot know which ones ever opened.
        if server.registry is not None:
            server.registry.close()
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='OpenAI-compatible server over a UniRT model')
    ap.add_argument(
        '--model',
        action='append',
        metavar='[NAME=]PATH',
        help='local model path or Hugging Face repository id. May be given '
             'more than once: the "model" field of a request then picks '
             'between them and /v1/models lists them all. NAME= sets the name '
             'clients use; without it the name is the path\'s last component. '
             'The first one is the default for requests that name no model, '
             'and is the only one loaded at startup -- the rest load when a '
             'request first asks for them',
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
    ap.add_argument(
        '--api-key',
        help='require Authorization: Bearer <key> on every /v1 request. Without '
             'it the server is open, which is fine on the default 127.0.0.1 bind '
             'and not fine with --host 0.0.0.0. UNIRT_API_KEY sets it too, which '
             'keeps the key out of the process list',
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
        '--slots',
        type=int,
        default=1,
        help='how many requests may decode at the same time. Each slot is a '
             'separate KV cache over the same weights (the plugin shares '
             'those), so the cost is one context per slot, not one model. '
             'Default 1, which is the old behaviour: a second request waits '
             'for the first to finish',
    )
    ap.add_argument(
        '--draft-model',
        help='a small model of the same vocabulary that proposes tokens for '
             '--model to verify in batches (speculative decoding). Same text '
             'either way -- the target keeps only what it agrees with -- so '
             'this is purely a latency bet, and it only pays when the draft '
             'is much cheaper than the target and agrees with it often. '
             'Applies to every --model given',
    )
    ap.add_argument(
        '--slot-timeout',
        type=float,
        default=120.0,
        help='seconds a request waits for a free slot before giving up with '
             '503 (default: 120)',
    )
    ap.add_argument(
        '--max-queued-requests',
        type=int,
        default=8,
        help='requests that may queue waiting for a decoding slot before the '
             'server starts returning 503. On top of the ones decoding, not '
             'including them (default: 8)',
    )
    ap.add_argument(
        '--max-resident-models',
        type=int,
        default=0,
        help='how many of the --model entries may be loaded at once; the '
             'least recently used idle one is closed to make room. 0 (the '
             'default) keeps every model that has been asked for, which is '
             'what fits when they were chosen to fit',
    )
    ap.add_argument(
        '--model-idle-timeout',
        type=float,
        default=0.0,
        help='seconds a loaded model may sit unused before it is closed and '
             'its memory given back; it reloads on the next request that '
             'names it. 0 (the default) never unloads',
    )
    args = ap.parse_args(argv)
    if args.n_ctx < 0:
        ap.error('--n-ctx must be >= 0')
    if args.max_resident_models < 0:
        ap.error('--max-resident-models must be >= 0')
    if args.model_idle_timeout < 0:
        ap.error('--model-idle-timeout must be >= 0')
    if args.max_queued_requests < 1:
        ap.error('--max-queued-requests must be >= 1')
    if args.slots < 1:
        ap.error('--slots must be >= 1')
    if args.slot_timeout <= 0:
        ap.error('--slot-timeout must be positive')
    api_key = args.api_key or os.environ.get('UNIRT_API_KEY') or None
    # Authorization travels as latin-1 and RFC 6750 bearer tokens are ASCII,
    # so a key outside ASCII is one no client can present -- the server would
    # start and then refuse everyone, which reads as a broken build rather
    # than a bad flag.
    if api_key is not None and not api_key.isascii():
        ap.error('--api-key must be ASCII; HTTP cannot carry anything else in a bearer token')
    if not args.model and not args.embedding_model and not args.rerank_model:
        ap.error('at least one of --model, --embedding-model or --rerank-model is required')

    def _source_and_id(path: str) -> tuple[str, str]:
        source = os.path.abspath(path) if os.path.exists(path) else path
        return source, os.path.splitext(os.path.basename(path.rstrip('/')))[0]

    def _spec(entry: str) -> ModelSpec:
        """`NAME=PATH` or just `PATH`, where the name is then the basename.

        Split on the first `=` only, and only when what precedes it is not
        itself a path -- Windows drive letters and `=` in a directory name are
        both rarer than getting this wrong would be annoying.
        """
        name, separator, path = entry.partition('=')
        if not separator or os.path.exists(entry) or '/' in name or os.sep in name:
            path, name = entry, ''
        source, derived = _source_and_id(path)
        return ModelSpec(
            name=name or derived,
            source=source,
            backend=args.backend,
            n_ctx=args.n_ctx,
            slots=args.slots,
            draft=args.draft_model,
        )

    registry = None
    model_id = None
    embedding = None
    embedding_id = None
    reranker = None
    reranker_id = None
    served: list[str] = []
    try:
        if args.model:
            specs = [_spec(entry) for entry in args.model]
            names = [spec.name for spec in specs]
            if len(set(names)) != len(names):
                ap.error(f'two models would answer to the same name: {", ".join(names)}')
            model_id = specs[0].name
            registry = ModelRegistry(
                specs,
                resident_limit=args.max_resident_models or None,
                idle_timeout=args.model_idle_timeout,
            )
            # Only the default loads now. A bad path or a missing plugin is
            # then reported while someone is still watching the terminal,
            # without paying for models this run may never use.
            print(f'loading {specs[0].source} on {args.backend} ...')
            entry = registry.preload(model_id)
            if entry.capabilities is not None:
                capabilities = ', '.join(
                    name for name, supported in entry.capabilities.items() if supported
                ) or 'no media modality'
                served.append(f'VLM: {capabilities}')
            else:
                stats = entry.model.runtime_stats()
                slots = f', {args.slots} slots' if args.slots > 1 else ''
                served.append(f"chat on {stats['device_name'] or '?'}{slots}")
            if len(specs) > 1:
                served.append(f'{len(specs)} models: {", ".join(names)} (loaded on demand)')

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

        if api_key:
            served.append('API key required')
        elif args.host not in ('127.0.0.1', 'localhost', '::1'):
            print(f'warning: listening on {args.host} with no --api-key -- '
                  'anything that can reach this port can use the model')
        print(f"ready on http://{args.host}:{args.port}/v1  ({'; '.join(served)})")
        serve(
            None,
            model_id,
            args.host,
            args.port,
            args.max_queued_requests,
            embedding,
            embedding_id,
            not args.no_prefix_cache,
            reranker,
            reranker_id,
            api_key,
            args.slot_timeout,
            registry,
        )
    finally:
        # serve() closes the registry, which owns whichever chat models ended
        # up loaded; it only reaches here if the server never started.
        if registry is not None:
            registry.close()
        for handle in (embedding, reranker):
            if handle is not None:
                handle.close()


if __name__ == '__main__':
    main()

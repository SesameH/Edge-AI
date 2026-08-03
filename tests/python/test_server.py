# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import unirt.server as server

from unirt.server import (
    _parse_generation_args,
    _prepare_messages,
    _validate_messages,
)


class FakeModel:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def _apply_chat_template(self, messages, *_):
        return messages[-1]['content']

    def generate(self, _prompt, **_kwargs):
        if self.fail:
            raise RuntimeError('boom')
        profile = SimpleNamespace(
            stop_reason='eos', prompt_tokens=2, generated_tokens=1
        )
        return SimpleNamespace(text='ok', profile=profile, logprobs=None)

    def runtime_stats(self):
        return {
            'model_bytes': 123,
            'kv_cache_bytes': 45,
            'device_peak_bytes': 678,
            'process_rss_bytes': 910,
            'device_name': 'Fake GPU',
        }


@pytest.mark.parametrize(
    'payload',
    [
        {'max_tokens': 0},
        {'temperature': float('nan')},
        {'top_p': 2},
        {'stop': [1]},
        {'max_tokens': 1.5},
        {'max_tokens': True},
        {'seed': 1.5},
    ],
)
def test_bad_generation_parameters_are_client_errors(payload):
    with pytest.raises(ValueError):
        _parse_generation_args(payload)


def test_text_server_rejects_malformed_or_multimodal_messages():
    with pytest.raises(ValueError, match='non-empty array'):
        _validate_messages([])
    with pytest.raises(ValueError, match='must be an object'):
        _validate_messages(['hello'])
    with pytest.raises(ValueError, match='text-only'):
        _validate_messages([{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}])


def test_message_normalization_accepts_null_content():
    assert _validate_messages([{'role': 'assistant', 'content': None}]) == [
        {'role': 'assistant', 'content': ''}
    ]


def test_vlm_request_materializes_and_cleans_inline_image():
    encoded = base64.b64encode(b'not-a-real-png-yet').decode()
    prepared = _prepare_messages(
        [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
                {'type': 'text', 'text': 'describe it'},
            ],
        }],
        multimodal=True,
    )
    path = prepared.images[0]
    assert os.path.isfile(path)
    assert prepared.messages[0]['content'][0] == {'type': 'image', 'image': path}
    prepared.cleanup()
    assert not os.path.exists(path)


def test_vlm_request_rejects_remote_media_urls():
    with pytest.raises(ValueError, match='inline'):
        _prepare_messages(
            [{
                'role': 'user',
                'content': [{
                    'type': 'image_url',
                    'image_url': {'url': 'https://example.invalid/image.png'},
                }],
            }],
            multimodal=True,
        )


def test_vlm_request_rejects_unsupported_loaded_modality_before_decoding():
    encoded = base64.b64encode(b'audio').decode()
    with pytest.raises(ValueError, match='does not support audio'):
        _prepare_messages(
            [{
                'role': 'user',
                'content': [{
                    'type': 'input_audio',
                    'input_audio': {'format': 'wav', 'data': encoded},
                }],
            }],
            multimodal=True,
            capabilities={'vision': True, 'audio': False},
        )


def test_server_main_accepts_hf_vlm_and_does_not_request_llm_stats(monkeypatch, capsys):
    class FakeVLM(server.UniRTVLM):
        def capabilities(self):
            return {'vision': True, 'audio': False}

        def close(self):
            self.closed = True

    model = object.__new__(FakeVLM)
    model.closed = False
    loaded = []
    served = []
    monkeypatch.setattr(
        server.AutoModelForCausalLM,
        'from_pretrained',
        lambda source, **kwargs: loaded.append((source, kwargs)) or model,
    )
    # *rest, **kwargs: this test is about which model main() loads and hands
    # over, not about serve()'s full signature, which keeps growing as
    # endpoints are added.
    def fake_serve(current, model_id, host, port, *rest, **kwargs):
        registry = kwargs.get('registry', rest[-1] if rest else None)
        # Read the registry here: main() closes it as it returns.
        served.append((registry.names(), [e.model for e in registry.loaded()],
                       model_id, host, port))

    monkeypatch.setattr(server, 'serve', fake_serve)
    monkeypatch.setattr(
        sys,
        'argv',
        ['unirt-server', '--model', 'acme/vision-GGUF', '--port', '9000'],
    )

    server.main()

    assert loaded == [
        ('acme/vision-GGUF', {'device_map': 'llama_cpp', 'n_ctx': 0, 'n_seq_max': 1})]
    # main() hands over a registry now, not handles: the default model is
    # loaded eagerly (once -- a VLM keeps media position state per handle, so
    # it gets one slot) and any others load on demand.
    names, resident, model_id, host, port = served[0]
    assert (model_id, host, port) == ('vision-GGUF', '127.0.0.1', 9000)
    assert names == ['vision-GGUF']
    assert resident == [model]
    assert model.closed
    assert 'VLM: vision' in capsys.readouterr().out


def test_cors_headers_for_browser_clients():
    """Regression for browser-based clients: preflight OPTIONS must succeed
    and every response must carry Access-Control-Allow-Origin."""
    import http.client

    httpd = server.UniRTHTTPServer(('127.0.0.1', 0), FakeModel(), 'fake-model')
    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request(
                'OPTIONS',
                '/v1/chat/completions',
                headers={
                    'Origin': 'http://localhost:3000',
                    'Access-Control-Request-Method': 'POST',
                },
            )
            preflight = connection.getresponse()
            preflight.read()
            assert preflight.status == 204
            assert preflight.getheader('Access-Control-Allow-Origin') == '*'
            assert 'POST' in (preflight.getheader('Access-Control-Allow-Methods') or '')

            connection.request('GET', '/v1/models', headers={'Origin': 'http://localhost:3000'})
            listing = connection.getresponse()
            listing.read()
            assert listing.status == 200
            assert listing.getheader('Access-Control-Allow-Origin') == '*'
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def test_stats_endpoint_exposes_runtime_stats():
    import http.client

    httpd = server.UniRTHTTPServer(('127.0.0.1', 0), FakeModel(), 'fake-model')
    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request('GET', '/v1/stats')
            response = connection.getresponse()
            body = json.loads(response.read())
            assert response.status == 200
            assert body['device_name'] == 'Fake GPU'
            assert body['model_bytes'] == 123
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def test_server_sheds_load_with_503_when_request_slots_exhausted():
    """request_slots bounds how many callers are in the server at once, so a
    burst gets a fast 503 instead of piling up blocked threads.

    The bound covers the requests that are decoding as well as the ones
    queued behind them: two slots plus a queue depth of one admits three.
    """
    import http.client

    httpd = server.UniRTHTTPServer(
        ('127.0.0.1', 0), [FakeModel(), FakeModel()], 'fake-model', max_queued_requests=1
    )
    admitted = 0
    while httpd.request_slots.acquire(blocking=False):
        admitted += 1
    assert admitted == 3  # 2 slots + 1 queued
    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            body = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
            connection.request(
                'POST', '/v1/chat/completions', body=body,
                headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            assert response.status == 503
            assert response.getheader('Retry-After') == '1'
            assert payload['error']['type'] == 'server_error'
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


TOOL_CALL_JSON = '{"name": "get_weather", "arguments": {"location": "Taipei"}}'


class FakeStreamer:
    """Emits the constrained output in pieces, like the real streamer."""

    def __init__(self, text: str, output, logprobs=None):
        self._pieces = [text[i:i + 8] for i in range(0, len(text), 8)]
        self.output = output
        # The real streamer exposes the live list only when logprobs were
        # asked for, and None otherwise.
        self.logprobs = logprobs

    def __iter__(self):
        return iter(self._pieces)

    def cancel(self):
        self._pieces = []


class FakeToolModel(FakeModel):
    """Stands in for a grammar-constrained model that decided to call a tool."""

    def generate(self, _prompt, *, stream: bool = False, **_kwargs):
        profile = SimpleNamespace(stop_reason='eos', prompt_tokens=2, generated_tokens=9)
        out = SimpleNamespace(text=TOOL_CALL_JSON, profile=profile, logprobs=None)
        return FakeStreamer(TOOL_CALL_JSON, out) if stream else out


def _post(httpd, payload: dict):
    import http.client

    connection = http.client.HTTPConnection('127.0.0.1', httpd.server_address[1], timeout=5)
    try:
        body = json.dumps(payload).encode()
        connection.request(
            'POST', '/v1/chat/completions', body=body,
            headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _serving(model):
    httpd = server.UniRTHTTPServer(('127.0.0.1', 0), model, 'fake-model')
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    return httpd, worker


WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'parameters': {
            'type': 'object',
            'properties': {'location': {'type': 'string'}},
            'required': ['location'],
        },
    },
}


def test_tool_call_response_matches_the_openai_shape():
    httpd, worker = _serving(FakeToolModel())
    try:
        status, payload = _post(httpd, {
            'messages': [{'role': 'user', 'content': 'weather in Taipei?'}],
            'tools': [WEATHER_TOOL],
            'tool_choice': 'required',
        })
        assert status == 200
        choice = payload['choices'][0]
        assert choice['finish_reason'] == 'tool_calls'
        assert choice['message']['content'] is None
        call = choice['message']['tool_calls'][0]
        assert call['function']['name'] == 'get_weather'
        assert json.loads(call['function']['arguments']) == {'location': 'Taipei'}
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def test_tools_and_response_format_cannot_share_the_grammar_slot():
    httpd, worker = _serving(FakeToolModel())
    try:
        status, payload = _post(httpd, {
            'messages': [{'role': 'user', 'content': 'hi'}],
            'tools': [WEATHER_TOOL],
            'response_format': {'type': 'json_object'},
        })
        assert status == 400
        assert 'cannot both constrain' in payload['error']['message']
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def test_streaming_tool_call_arrives_as_one_finished_delta():
    """Half a JSON envelope is useless to a client, so tool turns buffer."""
    import http.client

    httpd, worker = _serving(FakeToolModel())
    try:
        connection = http.client.HTTPConnection('127.0.0.1', httpd.server_address[1], timeout=5)
        try:
            body = json.dumps({
                'messages': [{'role': 'user', 'content': 'weather in Taipei?'}],
                'tools': [WEATHER_TOOL],
                'tool_choice': 'required',
                'stream': True,
            }).encode()
            connection.request(
                'POST', '/v1/chat/completions', body=body,
                headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
            )
            response = connection.getresponse()
            events = [
                json.loads(line[len('data: '):])
                for line in response.read().decode().splitlines()
                if line.startswith('data: ') and not line.endswith('[DONE]')
            ]
        finally:
            connection.close()
        deltas = [event['choices'][0]['delta'] for event in events]
        calls = [delta['tool_calls'] for delta in deltas if 'tool_calls' in delta]
        assert len(calls) == 1
        assert calls[0][0]['index'] == 0
        assert calls[0][0]['function']['name'] == 'get_weather'
        assert events[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


class FakeEmbedding:
    """Stands in for UniRTEmbedding: the server reaches for the same three
    members (_tokenize, _pad_id/_padding_side, encode_tokens)."""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self._pad_id = 0
        self._padding_side = 'right'
        self.seen: list[list[list[int]]] = []

    def _tokenize(self, texts):
        rows = [[1] * (len(text.split()) or 1) for text in texts]
        width = max(len(row) for row in rows)
        ids = [row + [self._pad_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return ids, masks, [[0] * width for _ in rows]

    def encode_tokens(self, input_ids, *, attention_mask=None, token_type_ids=None):
        self.seen.append(input_ids)
        return [
            [float(index) + offset / 10 for offset in range(self.dimension)]
            for index in range(len(input_ids))
        ]

    def runtime_stats(self):
        return {'device_name': 'Fake NPU'}


def _embedding_server(embedding=None, model=None):
    return server.UniRTHTTPServer(
        ('127.0.0.1', 0),
        model,
        'fake-model' if model is not None else None,
        8,
        embedding,
        'fake-embedding' if embedding is not None else None,
    )


def _post_to(httpd, path, payload):
    import http.client

    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            body = json.dumps(payload).encode()
            connection.request(
                'POST', path, body=body,
                headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def test_embeddings_returns_the_openai_shape():
    embedding = FakeEmbedding()
    status, body = _post_to(
        _embedding_server(embedding), '/v1/embeddings',
        {'model': 'fake-embedding', 'input': ['hello there', 'bye']},
    )
    assert status == 200
    assert body['object'] == 'list'
    assert body['model'] == 'fake-embedding'
    assert [entry['index'] for entry in body['data']] == [0, 1]
    assert all(entry['object'] == 'embedding' for entry in body['data'])
    assert body['data'][0]['embedding'] == [0.0, 0.1, 0.2, 0.3]
    # Padding must not be billed: 'hello there' is 2 tokens, 'bye' is 1.
    assert body['usage'] == {'prompt_tokens': 3, 'total_tokens': 3}


def test_embeddings_accepts_a_bare_string():
    status, body = _post_to(
        _embedding_server(FakeEmbedding()), '/v1/embeddings', {'input': 'one input'},
    )
    assert status == 200
    assert len(body['data']) == 1


def test_embeddings_base64_encoding_is_little_endian_float32():
    """The official OpenAI Python client asks for base64 whenever numpy is
    installed, so a float-only server silently breaks its default path."""
    import struct

    status, body = _post_to(
        _embedding_server(FakeEmbedding()), '/v1/embeddings',
        {'input': 'x', 'encoding_format': 'base64'},
    )
    assert status == 200
    packed = body['data'][0]['embedding']
    assert isinstance(packed, str)
    raw = base64.b64decode(packed)
    assert list(struct.unpack('<4f', raw)) == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_embeddings_accepts_pretokenized_input_and_pads_it():
    embedding = FakeEmbedding()
    status, body = _post_to(
        _embedding_server(embedding), '/v1/embeddings', {'input': [[5, 6, 7], [8]]},
    )
    assert status == 200
    assert len(body['data']) == 2
    # Ragged rows are padded into one rectangle for the native batch ABI.
    assert embedding.seen[-1] == [[5, 6, 7], [8, 0, 0]]
    assert body['usage']['prompt_tokens'] == 4


def test_embeddings_rejects_a_dimension_the_model_cannot_emit():
    status, body = _post_to(
        _embedding_server(FakeEmbedding()), '/v1/embeddings',
        {'input': 'x', 'dimensions': 512},
    )
    assert status == 400
    assert 'truncation' in body['error']['message']


@pytest.mark.parametrize('payload', [
    {},
    {'input': ''},
    {'input': []},
    {'input': [1, 'two']},
    {'input': 'x', 'encoding_format': 'binary'},
    {'input': 'x', 'dimensions': 0},
])
def test_embeddings_rejects_malformed_requests(payload):
    status, _ = _post_to(_embedding_server(FakeEmbedding()), '/v1/embeddings', payload)
    assert status == 400


def test_embeddings_without_an_embedding_model_is_a_clear_404():
    status, body = _post_to(
        _embedding_server(model=FakeModel()), '/v1/embeddings', {'input': 'x'},
    )
    assert status == 404
    assert '--embedding-model' in body['error']['message']


def test_chat_without_a_chat_model_is_a_clear_404():
    status, body = _post_to(
        _embedding_server(FakeEmbedding()), '/v1/chat/completions',
        {'messages': [{'role': 'user', 'content': 'hi'}]},
    )
    assert status == 404
    assert '--model' in body['error']['message']


def test_models_lists_both_when_both_are_loaded():
    import http.client

    httpd = _embedding_server(FakeEmbedding(), FakeModel())
    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request('GET', '/v1/models')
            body = json.loads(connection.getresponse().read())
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()
    assert [entry['id'] for entry in body['data']] == ['fake-model', 'fake-embedding']


def _chat(port, content: str, timeout: float = 60.0):
    import http.client

    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        body = json.dumps({
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': 8,
            'temperature': 0,
        }).encode()
        connection.request(
            'POST', '/v1/chat/completions', body=body,
            headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _stats(port):
    import http.client

    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        connection.request('GET', '/v1/stats')
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


def _run_server(model, reuse_prefix: bool, body):
    httpd = server.UniRTHTTPServer(
        ('127.0.0.1', 0), model, 'real-model', 8, None, None, reuse_prefix
    )
    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        return body(port)
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


@pytest.fixture()
def served_llm(sdk):
    """A model of this server's own, not the session-wide one: these tests
    deliberately leave KV state behind, which the shared fixture must not see."""
    from conftest import model_path
    from unirt.auto import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path('gguf'), device_map='llama_cpp')
    yield model
    model.close()


def test_cached_kv_survives_a_request_and_is_reused(served_llm):
    """End-to-end proof that the server stopped throwing the prefix cache away:
    kv_cache_bytes is near-empty on a fresh model, and stays populated after a
    completed request so the next one can match a prefix against it."""

    def body(port):
        empty = _stats(port)['kv_cache_bytes']
        status, _ = _chat(port, 'Name three colours.')
        assert status == 200
        after_first = _stats(port)['kv_cache_bytes']

        # A second turn that extends the first must still succeed, and leave
        # cached state behind in turn.
        status, payload = _chat(port, 'Name three colours. Then name three fruits.')
        assert status == 200
        assert payload['choices'][0]['message']['content'] is not None
        return empty, after_first, _stats(port)['kv_cache_bytes']

    empty, after_first, after_second = _run_server(served_llm, True, body)
    assert after_first > empty
    assert after_second > empty


def test_no_prefix_cache_clears_kv_after_every_request(served_llm):
    def body(port):
        empty = _stats(port)['kv_cache_bytes']
        status, _ = _chat(port, 'Name three colours.')
        assert status == 200
        return empty, _stats(port)['kv_cache_bytes']

    empty, after = _run_server(served_llm, False, body)
    assert after == empty


def test_failed_generation_leaves_no_cached_state(served_llm):
    """The failure path must still reset: a run that raised may leave the
    plugin's token transcript out of step with what is actually in KV."""
    original = served_llm.generate
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('boom')

    def body(port):
        empty = _stats(port)['kv_cache_bytes']
        served_llm.generate = failing
        try:
            status, payload = _chat(port, 'this one explodes')
        finally:
            served_llm.generate = original
        assert status == 500
        assert 'boom' in payload['error']['message']
        return empty, _stats(port)['kv_cache_bytes']

    empty, after = _run_server(served_llm, True, body)
    assert calls == [1]
    assert after == empty


class FakeReranker:
    """Stands in for a UniRTEmbedding opened on a cross-encoder: rerank() only."""

    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        if self.scores is not None:
            return list(self.scores)
        # Deterministic and order-independent: longer documents score higher.
        return [float(len(text)) for text in documents]

    def runtime_stats(self):
        return {'device_name': 'Fake NPU'}


def _rerank_server(reranker):
    return server.UniRTHTTPServer(
        ('127.0.0.1', 0), None, None, 8, None, None, True, reranker, 'fake-reranker'
    )


def test_rerank_sorts_by_score_and_keeps_original_indices():
    reranker = FakeReranker(scores=[-2.0, 5.0, 1.0])
    status, body = _post_to(
        _rerank_server(reranker), '/v1/rerank',
        {'query': 'q', 'documents': ['first', 'second', 'third']},
    )
    assert status == 200
    assert body['object'] == 'list'
    assert body['model'] == 'fake-reranker'
    # Sorted best first, but index still points into the request's own array.
    assert [entry['index'] for entry in body['results']] == [1, 2, 0]
    scores = [entry['relevance_score'] for entry in body['results']]
    assert scores == sorted(scores, reverse=True)
    assert reranker.calls == [('q', ['first', 'second', 'third'])]


def test_rerank_maps_logits_into_zero_to_one():
    """relevance_score means 0..1 to every client that speaks this API; the
    model emits unbounded cross-encoder logits."""
    status, body = _post_to(
        _rerank_server(FakeReranker(scores=[0.0, 8.6, -11.0, -800.0])), '/v1/rerank',
        {'query': 'q', 'documents': ['a', 'b', 'c', 'd']},
    )
    assert status == 200
    by_index = {entry['index']: entry['relevance_score'] for entry in body['results']}
    assert by_index[0] == pytest.approx(0.5)
    assert by_index[1] == pytest.approx(1 / (1 + 2.718281828 ** -8.6), rel=1e-6)
    assert 0.0 < by_index[2] < 0.001
    # A very negative logit must not overflow on the way to a probability.
    assert by_index[3] == pytest.approx(0.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in by_index.values())


def test_rerank_top_n_truncates_after_sorting():
    status, body = _post_to(
        _rerank_server(FakeReranker(scores=[1.0, 9.0, 5.0])), '/v1/rerank',
        {'query': 'q', 'documents': ['a', 'b', 'c'], 'top_n': 2},
    )
    assert status == 200
    assert [entry['index'] for entry in body['results']] == [1, 2]


def test_rerank_returns_documents_only_when_asked():
    payload = {'query': 'q', 'documents': ['alpha', 'beta']}
    status, without = _post_to(_rerank_server(FakeReranker()), '/v1/rerank', payload)
    assert status == 200
    assert all('document' not in entry for entry in without['results'])

    status, with_documents = _post_to(
        _rerank_server(FakeReranker()), '/v1/rerank', {**payload, 'return_documents': True},
    )
    assert status == 200
    # The text must follow its own result, not the request order.
    for entry in with_documents['results']:
        assert entry['document']['text'] == payload['documents'][entry['index']]


def test_rerank_accepts_cohere_style_document_objects():
    reranker = FakeReranker()
    status, _ = _post_to(
        _rerank_server(reranker), '/v1/rerank',
        {'query': 'q', 'documents': [{'text': 'alpha'}, {'text': 'beta'}]},
    )
    assert status == 200
    assert reranker.calls == [('q', ['alpha', 'beta'])]


@pytest.mark.parametrize('payload', [
    {'documents': ['a']},
    {'query': '', 'documents': ['a']},
    {'query': 'q'},
    {'query': 'q', 'documents': []},
    {'query': 'q', 'documents': [1]},
    {'query': 'q', 'documents': [{'text': ''}]},
    {'query': 'q', 'documents': ['a'], 'top_n': 0},
    {'query': 'q', 'documents': ['a'], 'top_n': True},
    {'query': 'q', 'documents': ['a'], 'return_documents': 'yes'},
])
def test_rerank_rejects_malformed_requests(payload):
    status, _ = _post_to(_rerank_server(FakeReranker()), '/v1/rerank', payload)
    assert status == 400


def test_rerank_without_a_reranker_is_a_clear_404():
    status, body = _post_to(
        _embedding_server(FakeEmbedding()), '/v1/rerank', {'query': 'q', 'documents': ['a']},
    )
    assert status == 404
    assert '--rerank-model' in body['error']['message']


def test_rerank_reports_a_model_failure_as_a_server_error():
    class Broken(FakeReranker):
        def rerank(self, query, documents):
            raise RuntimeError('no classifier head')

    status, body = _post_to(
        _rerank_server(Broken()), '/v1/rerank', {'query': 'q', 'documents': ['a']},
    )
    assert status == 500
    assert 'no classifier head' in body['error']['message']


def _real_reranker():
    from conftest import REPO_ROOT
    from unirt import AutoModelForEmbedding

    bundle = os.path.join(REPO_ROOT, 'models', 'bge-reranker-v2-m3-GGUF')
    if not os.path.isdir(bundle) or not any(
        name.endswith('.gguf') for name in os.listdir(bundle)
    ):
        pytest.skip('GGUF reranker model not downloaded (see README)')
    return AutoModelForEmbedding.from_pretrained(bundle)


def test_rerank_ranks_a_real_corpus(sdk):
    """The public loader must open a rerank-only GGUF (no tokenizer.json), and
    the endpoint must put the relevant document first."""
    model = _real_reranker()
    assert model._tokenizer is None
    try:
        status, body = _post_to(
            _rerank_server(model), '/v1/rerank',
            {
                'query': 'What is the capital of France?',
                'documents': [
                    'Kangaroos are marsupials native to Australia.',
                    'Paris is the capital and largest city of France.',
                    'The Loire is the longest river in France.',
                ],
                'return_documents': True,
            },
        )
    finally:
        model.close()
    assert status == 200
    assert body['results'][0]['index'] == 1
    assert 'Paris' in body['results'][0]['document']['text']
    assert body['results'][0]['relevance_score'] > body['results'][-1]['relevance_score']
    assert all(0.0 <= entry['relevance_score'] <= 1.0 for entry in body['results'])


@pytest.mark.parametrize(('field', 'value'), [
    ('top_k', 40),
    ('min_p', 0.05),
    ('repetition_penalty', 1.1),
    ('presence_penalty', 0.6),
    ('frequency_penalty', -0.4),
])
def test_sampler_parameters_reach_the_model(field, value):
    """Accepting a sampling parameter and then dropping it looks exactly like
    the parameter having no effect on the model."""
    assert _parse_generation_args({field: value})[field] == value


def test_sampler_parameters_are_absent_when_not_requested():
    """Anything not sent must stay unset so the plugin's own default applies,
    rather than being pinned to a zero the caller never asked for."""
    parsed = _parse_generation_args({})
    for field in ('top_k', 'min_p', 'repetition_penalty',
                  'presence_penalty', 'frequency_penalty'):
        assert field not in parsed


@pytest.mark.parametrize('payload', [
    {'top_k': -1},
    {'top_k': 1.5},
    {'top_k': True},
    {'min_p': 1.5},
    {'min_p': float('nan')},
    {'repetition_penalty': -1},
    {'presence_penalty': 3},
    {'frequency_penalty': -3},
    {'frequency_penalty': 'high'},
])
def test_bad_sampler_parameters_are_client_errors(payload):
    with pytest.raises(ValueError):
        _parse_generation_args(payload)


def test_sampler_parameters_survive_the_http_round_trip():
    """The parser is only half of it -- the values have to arrive at generate()."""
    class Recorder(FakeModel):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def generate(self, prompt, **kwargs):
            self.kwargs = kwargs
            return super().generate(prompt, **{})

    model = Recorder()
    status, _ = _post_to(
        server.UniRTHTTPServer(('127.0.0.1', 0), model, 'fake-model'),
        '/v1/chat/completions',
        {
            'messages': [{'role': 'user', 'content': 'hi'}],
            'top_k': 40, 'min_p': 0.05, 'repetition_penalty': 1.1,
            'presence_penalty': 0.6, 'frequency_penalty': -0.4,
        },
    )
    assert status == 200
    assert model.kwargs['top_k'] == 40
    assert model.kwargs['min_p'] == pytest.approx(0.05)
    assert model.kwargs['repetition_penalty'] == pytest.approx(1.1)
    assert model.kwargs['presence_penalty'] == pytest.approx(0.6)
    assert model.kwargs['frequency_penalty'] == pytest.approx(-0.4)


def _keyed_server(api_key):
    return server.UniRTHTTPServer(
        ('127.0.0.1', 0), FakeModel(), 'fake-model', 8, None, None, True, None, None, api_key
    )


def _request(httpd, method, path, headers=None, payload=None):
    import http.client

    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            sent = dict(headers or {})
            if body is not None:
                sent.update({'Content-Type': 'application/json',
                             'Content-Length': str(len(body))})
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            raw = response.read()
            return response.status, response.getheader('WWW-Authenticate'), raw
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


_CHAT = {'messages': [{'role': 'user', 'content': 'hi'}]}


@pytest.mark.parametrize(('method', 'path', 'payload'), [
    ('GET', '/v1/models', None),
    ('GET', '/v1/stats', None),
    ('POST', '/v1/chat/completions', _CHAT),
])
def test_api_key_is_required_on_every_v1_endpoint(method, path, payload):
    status, challenge, _ = _request(_keyed_server('s3cret'), method, path, payload=payload)
    assert status == 401
    assert challenge == 'Bearer'


def test_api_key_admits_the_right_bearer_token():
    status, _, raw = _request(
        _keyed_server('s3cret'), 'POST', '/v1/chat/completions',
        headers={'Authorization': 'Bearer s3cret'}, payload=_CHAT,
    )
    assert status == 200
    assert json.loads(raw)['choices'][0]['message']['content'] == 'ok'


@pytest.mark.parametrize('header', [
    'Bearer wrong',
    'Bearer s3cre',          # a prefix of the real key
    'Bearer s3cretX',
    'Basic s3cret',          # right secret, wrong scheme
    's3cret',                # no scheme
    '',
])
def test_api_key_rejects_anything_but_an_exact_bearer_match(header):
    status, _, _ = _request(
        _keyed_server('s3cret'), 'POST', '/v1/chat/completions',
        headers={'Authorization': header}, payload=_CHAT,
    )
    assert status == 401


def test_a_non_ascii_bearer_token_is_rejected_not_fatal():
    """compare_digest raises TypeError on str holding non-ASCII rather than
    returning False, so an unauthenticated caller could kill the request
    thread at will by sending one byte over 0x7f."""
    status, _, _ = _request(
        _keyed_server('s3cret'), 'POST', '/v1/chat/completions',
        headers={'Authorization': 'Bearer s3crét'}, payload=_CHAT,
    )
    assert status == 401


def test_a_non_ascii_configured_key_rejects_instead_of_crashing():
    """The same defect fired from the other side: a key the operator chose with
    a non-ASCII character made compare_digest raise on every request, valid
    attempt or not. main() refuses such a key up front, but the handler must
    not depend on that."""
    status, _, _ = _request(
        _keyed_server('s3crét'), 'POST', '/v1/chat/completions',
        headers={'Authorization': 'Bearer s3cret'}, payload=_CHAT,
    )
    assert status == 401


def test_main_refuses_an_api_key_no_client_could_send():
    """Authorization is carried as latin-1 and bearer tokens are ASCII by
    spec, so a key outside ASCII is one nothing can ever authenticate with --
    a server that starts anyway is simply unreachable."""
    with pytest.raises(SystemExit):
        server.main(['--model', 'unused', '--api-key', 'парол'])


def test_health_stays_open_so_probes_do_not_need_the_key():
    """A container healthcheck should not have to carry the secret, and the
    reply discloses nothing beyond the fact that something is listening."""
    status, _, raw = _request(_keyed_server('s3cret'), 'GET', '/health')
    assert status == 200
    assert json.loads(raw)['status'] == 'ok'


def test_no_api_key_configured_leaves_the_server_open():
    status, _, _ = _request(_keyed_server(None), 'GET', '/v1/models')
    assert status == 200


# ---------------------------------------------------------------------------
# /v1/completions -- the pre-chat endpoint
# ---------------------------------------------------------------------------


class EchoModel(FakeModel):
    """Records the prompt it was handed, so tests can prove no template ran."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def generate(self, prompt, *, stream: bool = False, **_kwargs):
        self.prompts.append(prompt)
        text = f'<{prompt}>'
        profile = SimpleNamespace(stop_reason='eos', prompt_tokens=3, generated_tokens=2)
        out = SimpleNamespace(text=text, profile=profile, logprobs=None)
        return FakeStreamer(text, out) if stream else out


def _completion_server(model):
    return server.UniRTHTTPServer(('127.0.0.1', 0), model, 'fake-model')


def test_completions_sends_the_prompt_through_untouched():
    """No chat template: that is the reason this endpoint exists next to
    /v1/chat/completions, and a base model has no template to apply anyway."""
    model = EchoModel()
    status, body = _post_to(
        _completion_server(model), '/v1/completions', {'prompt': 'Once upon a'}
    )
    assert status == 200
    assert model.prompts == ['Once upon a']
    assert body['object'] == 'text_completion'
    assert body['id'].startswith('cmpl-')
    assert body['choices'] == [
        {'index': 0, 'text': '<Once upon a>', 'logprobs': None, 'finish_reason': 'stop'}
    ]
    assert body['usage'] == {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}


def test_completions_echo_prepends_the_prompt():
    status, body = _post_to(
        _completion_server(EchoModel()),
        '/v1/completions',
        {'prompt': 'seed', 'echo': True},
    )
    assert status == 200
    assert body['choices'][0]['text'] == 'seed<seed>'


def test_completions_runs_every_prompt_in_an_array():
    model = EchoModel()
    status, body = _post_to(
        _completion_server(model), '/v1/completions', {'prompt': ['a', 'b', 'c']}
    )
    assert status == 200
    assert model.prompts == ['a', 'b', 'c']
    assert [choice['index'] for choice in body['choices']] == [0, 1, 2]
    assert [choice['text'] for choice in body['choices']] == ['<a>', '<b>', '<c>']
    # Usage is the whole request, not the last prompt in it.
    assert body['usage']['total_tokens'] == 15


@pytest.mark.parametrize('payload', [
    {},                                     # no prompt at all
    {'prompt': ''},                         # empty is not a prompt
    {'prompt': []},
    {'prompt': [1, 2, 3]},                  # pre-tokenized: generate() takes text
    {'prompt': 'hi', 'n': 2},
    {'prompt': 'hi', 'best_of': 4},
    {'prompt': 'hi', 'suffix': 'tail'},     # fill-in-the-middle
    {'prompt': 'hi', 'logprobs': 21},       # above the cap
    {'prompt': 'hi', 'logprobs': 'yes'},
    {'prompt': 'hi', 'logprobs': -1},
    {'prompt': 'hi', 'echo': 'yes'},
    {'prompt': 'hi', 'stream': 'yes'},
    {'prompt': 'hi\x00there'},
])
def test_completions_rejects_what_it_cannot_honour(payload):
    """Accepting a parameter and quietly ignoring it is worse than refusing:
    the reply would look correct while meaning something else."""
    status, body = _post_to(_completion_server(EchoModel()), '/v1/completions', payload)
    assert status == 400, body


def test_completions_needs_a_chat_model():
    status, body = _post_to(_completion_server(None), '/v1/completions', {'prompt': 'hi'})
    assert status == 404
    assert '--model' in body['error']['message']


def _post_raw(httpd, path, payload):
    """POST and return the raw body, for streaming responses."""
    import http.client

    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            body = json.dumps(payload).encode()
            connection.request(
                'POST', path, body=body,
                headers={'Content-Type': 'application/json', 'Content-Length': str(len(body))},
            )
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()


def _sse_events(raw: str) -> list[dict]:
    return [
        json.loads(line[len('data: '):])
        for line in raw.splitlines()
        if line.startswith('data: ') and line != 'data: [DONE]'
    ]


def test_completions_streaming_matches_the_blocking_text():
    status, raw = _post_raw(
        _completion_server(EchoModel()),
        '/v1/completions',
        {'prompt': 'stream me', 'stream': True},
    )
    assert status == 200
    assert raw.rstrip().endswith('data: [DONE]')
    events = _sse_events(raw)
    assert all(event['object'] == 'text_completion' for event in events)
    assert ''.join(event['choices'][0]['text'] for event in events) == '<stream me>'
    # Exactly one terminating chunk, and it is the last.
    finishes = [event['choices'][0]['finish_reason'] for event in events]
    assert finishes[-1] == 'stop'
    assert finishes.count('stop') == 1


def test_completions_streaming_keeps_each_prompt_on_its_own_index():
    status, raw = _post_raw(
        _completion_server(EchoModel()),
        '/v1/completions',
        {'prompt': ['one', 'two'], 'stream': True},
    )
    assert status == 200
    events = _sse_events(raw)
    by_index: dict[int, str] = {}
    for event in events:
        choice = event['choices'][0]
        by_index[choice['index']] = by_index.get(choice['index'], '') + choice['text']
    assert by_index == {0: '<one>', 1: '<two>'}


# ---------------------------------------------------------------------------
# logprobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('payload', 'expected'), [
    ({}, 0),
    ({'logprobs': False}, 0),
    # The native side always reports the token it sampled, so "logprobs on,
    # no alternatives" is still a request for one entry.
    ({'logprobs': True}, 1),
    ({'logprobs': True, 'top_logprobs': 0}, 1),
    ({'logprobs': True, 'top_logprobs': 5}, 5),
])
def test_chat_logprobs_request_shapes(payload, expected):
    assert server._parse_logprobs_request(payload, chat=True) == expected


@pytest.mark.parametrize('payload', [
    {'logprobs': 3},                      # chat spells it as a boolean
    {'top_logprobs': 3},                  # without logprobs: true
    {'logprobs': True, 'top_logprobs': 21},
    {'logprobs': True, 'top_logprobs': -1},
    {'logprobs': True, 'top_logprobs': 'many'},
])
def test_chat_logprobs_request_rejects_bad_shapes(payload):
    with pytest.raises(ValueError):
        server._parse_logprobs_request(payload, chat=True)


@pytest.mark.parametrize(('payload', 'expected'), [
    ({}, 0),
    ({'logprobs': 0}, 1),                 # the older endpoint spells it as a count
    ({'logprobs': 4}, 4),
])
def test_completion_logprobs_request_shapes(payload, expected):
    assert server._parse_logprobs_request(payload, chat=False) == expected


def _fake_step(token: str, token_id: int, logprob: float, top=()):
    from unirt.generation.output import Logprob, TokenLogprobs

    return TokenLogprobs(
        chosen=Logprob(token=token, token_id=token_id, logprob=logprob),
        top=tuple(Logprob(token=t, token_id=i, logprob=p) for t, i, p in top),
    )


def test_logprob_payload_matches_the_openai_shape():
    payload = server._logprob_payload([
        _fake_step('Hi', 42, -0.25, top=[('Hi', 42, -0.25), ('Hey', 43, -1.5)]),
    ])
    entry = payload['content'][0]
    assert entry['token'] == 'Hi'
    assert entry['logprob'] == -0.25
    # bytes let a client rebuild a character a single token only carries part
    # of, which `token` alone cannot express.
    assert entry['bytes'] == [72, 105]
    assert [alt['token'] for alt in entry['top_logprobs']] == ['Hi', 'Hey']


def test_logprob_bytes_survive_a_token_that_is_not_whole_text():
    """Tokens split multi-byte characters routinely; the bytes field is the
    only part of the response that stays faithful when that happens."""
    payload = server._logprob_payload([_fake_step('�', 7, -2.0)])
    assert payload['content'][0]['bytes'] == list('�'.encode('utf-8'))


def test_pending_logprobs_attaches_each_token_to_the_chunk_it_produced():
    """A chunk is not a token: the bridge holds a piece back until the bytes
    that finish its character arrive, so one chunk can cover several tokens."""
    streamer = SimpleNamespace(logprobs=[
        _fake_step('a', 1, -0.1), _fake_step('b', 2, -0.2), _fake_step('c', 3, -0.3),
    ])
    first = server._pending_logprobs(streamer, 0)
    assert [entry['token'] for entry in first['content']] == ['a', 'b', 'c']
    # Nothing new since: no logprobs block rather than a repeat of the last.
    assert server._pending_logprobs(streamer, 3) is None
    assert [e['token'] for e in server._pending_logprobs(streamer, 2)['content']] == ['c']


def test_no_logprobs_asked_for_means_no_logprobs_block():
    assert server._pending_logprobs(SimpleNamespace(logprobs=None), 0) is None


def test_the_two_endpoints_use_different_logprob_shapes():
    """Not an accident to paper over: /v1/completions predates the per-token
    object list and carries four parallel arrays instead. A client built on
    the official schema rejects the chat shape there outright."""
    steps = [
        _fake_step(' Paris', 10, -0.76, top=[(' the', 11, -1.33)]),
        _fake_step('.', 12, -0.62),
    ]
    chat = server._logprob_payload(steps)
    legacy = server._legacy_logprob_payload(steps, 0)

    assert set(chat) == {'content'}
    assert set(legacy) == {'tokens', 'token_logprobs', 'top_logprobs', 'text_offset'}
    assert legacy['tokens'] == [' Paris', '.']
    assert legacy['token_logprobs'] == [-0.76, -0.62]
    # A mapping per position, not a list of objects.
    assert legacy['top_logprobs'] == [{' the': -1.33}, {}]


class FakeTrimmedStreamer:
    """A run whose last decoded token contributes no text.

    That is what a stop sequence looks like from here: the token that
    triggered it is decoded, reported, and then trimmed out of the answer. The
    steps list therefore outgrows the pieces the client ever sees.
    """

    def __init__(self, output):
        self.output = output
        self.logprobs = []

    def __iter__(self):
        for piece in ('Hel', 'lo'):
            self.logprobs.append(_fake_step(piece, len(self.logprobs), -0.5))
            yield piece
        self.logprobs.append(_fake_step(' STOP', 99, -1.5))

    def cancel(self):
        pass


class FakeTrimmedModel(FakeModel):
    def generate(self, _prompt, *, stream: bool = False, **_kwargs):
        profile = SimpleNamespace(stop_reason='stop_sequence', prompt_tokens=2,
                                  generated_tokens=3)
        steps = [_fake_step('Hel', 0, -0.5), _fake_step('lo', 1, -0.5),
                 _fake_step(' STOP', 99, -1.5)]
        out = SimpleNamespace(text='Hello', profile=profile, logprobs=steps)
        return FakeTrimmedStreamer(out) if stream else out


def test_streaming_reports_every_logprob_the_blocking_reply_would():
    """Otherwise the same request answers with a different number of tokens
    depending only on whether it was streamed."""
    request = {'messages': [{'role': 'user', 'content': 'hi'}],
               'logprobs': True, 'top_logprobs': 0}
    _, blocking = _post_to(
        _completion_server(FakeTrimmedModel()), '/v1/chat/completions', dict(request))
    _, raw = _post_raw(
        _completion_server(FakeTrimmedModel()), '/v1/chat/completions',
        dict(request, stream=True))
    chunks = _sse_events(raw)

    expected = [entry['token'] for entry in blocking['choices'][0]['logprobs']['content']]
    streamed = [
        entry['token']
        for chunk in chunks
        for entry in (chunk['choices'][0].get('logprobs') or {}).get('content', [])
    ]
    assert expected == ['Hel', 'lo', ' STOP']
    assert streamed == expected
    # The trimmed token rides on the chunk that closes the stream, since no
    # text chunk followed it.
    final = chunks[-1]['choices'][0]
    assert final['finish_reason'] == 'stop'
    assert [e['token'] for e in final['logprobs']['content']] == [' STOP']


def test_legacy_text_offsets_track_the_returned_text():
    steps = [_fake_step('ab', 1, -0.1), _fake_step('cde', 2, -0.2), _fake_step('f', 3, -0.3)]
    assert server._legacy_logprob_payload(steps, 0)['text_offset'] == [0, 2, 5]
    # echo puts the prompt in front of the completion, so every token moves.
    assert server._legacy_logprob_payload(steps, 7)['text_offset'] == [7, 9, 12]


# ---------------------------------------------------------------------------
# SlotPool
# ---------------------------------------------------------------------------


class SlotModel:
    """Minimal stand-in that records resets, for pool bookkeeping tests."""

    def __init__(self, name: str):
        self.name = name
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def _pool(count: int):
    return server.SlotPool([SlotModel(f'm{index}') for index in range(count)])


def test_a_new_conversation_takes_an_idle_slot_rather_than_evicting_a_live_one():
    """The bug this pins: every conversation key starts with the same role
    marker, so scoring by longest common prefix made an unrelated request look
    like a better match for a busy slot than an untouched one was."""
    pool = _pool(3)
    alpha = server._conversation_key([{'role': 'user', 'content': 'Alpha alpha.'}])
    beta = server._conversation_key([{'role': 'user', 'content': 'Beta beta.'}])

    with pool.checkout(alpha) as first:
        first_model = first.model
    with pool.checkout(beta) as second:
        assert second.model is not first_model


def test_a_continuing_conversation_returns_to_its_own_slot():
    pool = _pool(3)
    first_turn = [{'role': 'user', 'content': 'Alpha alpha.'}]
    with pool.checkout(server._conversation_key(first_turn)) as slot:
        home = slot.model
    with pool.checkout(server._conversation_key([{'role': 'user', 'content': 'Beta.'}])):
        pass

    later = first_turn + [
        {'role': 'assistant', 'content': 'hi'},
        {'role': 'user', 'content': 'And again.'},
    ]
    with pool.checkout(server._conversation_key(later)) as slot:
        assert slot.model is home


def test_a_conversation_key_grows_by_appending():
    """Slot affinity is a startswith test, so a turn must extend the key its
    predecessor stored rather than rewrite it."""
    messages = [{'role': 'user', 'content': 'one'}]
    first = server._conversation_key(messages)
    messages += [{'role': 'assistant', 'content': 'two'}, {'role': 'user', 'content': 'three'}]
    assert server._conversation_key(messages).startswith(first)


def test_multimodal_content_still_produces_a_key():
    key = server._conversation_key([
        {'role': 'user', 'content': [
            {'type': 'image', 'image': '/tmp/x.png'},
            {'type': 'text', 'text': 'describe'},
        ]},
    ])
    assert 'describe' in key


def test_slots_are_handed_out_one_at_a_time_and_returned():
    pool = _pool(2)
    with pool.checkout('a'):
        with pool.checkout('b'):
            # Both taken; a third caller must wait rather than share one.
            with pytest.raises(TimeoutError):
                with pool.checkout('c', timeout=0.05):
                    pass
    # Both back.
    with pool.checkout('a'), pool.checkout('b'):
        pass


def test_a_failed_run_drops_that_slot_s_cache_but_keeps_the_slot():
    pool = _pool(2)
    with pytest.raises(RuntimeError):
        with pool.checkout('a') as slot:
            failed = slot.model
            raise RuntimeError('boom')
    assert failed.resets == 1
    # Reusable straight away, and no longer claiming to hold that conversation.
    with pool.checkout('a') as slot:
        assert slot.affinity == ''


def test_prefix_reuse_can_be_turned_off_per_pool():
    pool = _pool(1)
    with pool.checkout('a', reuse_prefix=False) as slot:
        model = slot.model
    assert model.resets == 1
    with pool.checkout('a') as slot:
        assert slot.affinity == ''


def test_a_successful_run_keeps_the_cache():
    pool = _pool(1)
    with pool.checkout('a') as slot:
        model = slot.model
    assert model.resets == 0


def test_concurrent_checkouts_never_hand_the_same_slot_to_two_callers():
    pool = _pool(4)
    overlapping = []
    live = set()
    guard = threading.Lock()

    def worker(index):
        with pool.checkout(f'conversation-{index}') as slot:
            with guard:
                if slot.model in live:
                    overlapping.append(slot.model)
                live.add(slot.model)
            time.sleep(0.01)
            with guard:
                live.discard(slot.model)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlapping


def test_closing_the_pool_closes_every_handle():
    pool = _pool(3)
    pool.close()
    assert all(model.closed for model in pool.models)


def test_a_pool_needs_at_least_one_handle():
    with pytest.raises(ValueError):
        server.SlotPool([])


# ---------------------------------------------------------------------------
# ModelRegistry -- several named models, loaded on demand
# ---------------------------------------------------------------------------


class NamedModel(FakeModel):
    """A chat model whose answer says which model answered."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.closed = False

    def generate(self, _prompt, **_kwargs):
        profile = SimpleNamespace(stop_reason='eos', prompt_tokens=2, generated_tokens=1)
        return SimpleNamespace(text=f'answered by {self.name}', profile=profile, logprobs=None)

    def close(self):
        self.closed = True


def _registry(names, **kwargs):
    """A registry over models that count how often they were loaded."""
    loads: dict[str, int] = {name: 0 for name in names}
    opened: dict[str, list] = {name: [] for name in names}

    def loader(spec):
        loads[spec.name] += 1
        handles = [NamedModel(spec.name) for _ in range(spec.slots)]
        opened[spec.name].extend(handles)
        return handles

    specs = [server.ModelSpec(name=name, source=f'/models/{name}') for name in names]
    return server.ModelRegistry(specs, loader=loader, **kwargs), loads, opened


def test_a_model_loads_when_it_is_first_named_and_not_before():
    registry, loads, _ = _registry(['alpha', 'beta'])
    try:
        assert loads == {'alpha': 0, 'beta': 0}
        with registry.acquire('beta') as entry:
            assert entry.name == 'beta'
        assert loads == {'alpha': 0, 'beta': 1}
        # Still resident, so naming it again is free.
        with registry.acquire('beta'):
            pass
        assert loads['beta'] == 1
    finally:
        registry.close()


def test_an_unknown_model_is_an_error_only_when_there_is_a_choice():
    single, _, _ = _registry(['alpha'])
    several, _, _ = _registry(['alpha', 'beta'])
    try:
        # One model: the name cannot route anywhere else, and clients that
        # hardcode an OpenAI name would break for no gain.
        assert single.resolve('gpt-4') == 'alpha'
        assert single.resolve(None) == 'alpha'
        # Several: answering from the wrong one is the failure to avoid.
        assert several.resolve(None) == 'alpha'
        assert several.resolve('beta') == 'beta'
        with pytest.raises(KeyError):
            several.resolve('gpt-4')
        with pytest.raises(ValueError):
            several.resolve(7)
    finally:
        single.close()
        several.close()


def test_the_least_recently_idle_model_is_evicted_over_the_limit():
    registry, loads, opened = _registry(['alpha', 'beta', 'gamma'], resident_limit=2)
    try:
        for name in ('alpha', 'beta'):
            with registry.acquire(name):
                pass
        with registry.acquire('gamma'):
            pass
        assert sorted(entry.name for entry in registry.loaded()) == ['beta', 'gamma']
        assert opened['alpha'][0].closed
        # Asking for it again reloads it, and pushes out the next-oldest.
        with registry.acquire('alpha'):
            pass
        assert loads['alpha'] == 2
        assert sorted(entry.name for entry in registry.loaded()) == ['alpha', 'gamma']
    finally:
        registry.close()


def test_a_model_in_use_is_never_evicted_out_from_under_the_request():
    registry, _, opened = _registry(['alpha', 'beta'], resident_limit=1)
    try:
        with registry.acquire('alpha'):
            with registry.acquire('beta'):
                # Over the limit, but neither is idle: staying over beats
                # closing a model a live request is decoding on.
                assert len(registry.loaded()) == 2
                assert not opened['alpha'][0].closed
        # Both idle now; the next load brings it back to the limit.
        with registry.acquire('beta'):
            pass
        assert [entry.name for entry in registry.loaded()] == ['beta']
        assert opened['alpha'][0].closed
    finally:
        registry.close()


def test_an_idle_model_is_given_back_after_the_idle_timeout():
    registry, loads, opened = _registry(['alpha'], idle_timeout=0.05)
    try:
        with registry.acquire('alpha'):
            pass
        deadline = time.monotonic() + 5
        while registry.loaded() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not registry.loaded(), 'the reaper never closed the idle model'
        assert opened['alpha'][0].closed
        # It comes back on the next request that names it.
        with registry.acquire('alpha'):
            pass
        assert loads['alpha'] == 2
    finally:
        registry.close()


def test_closing_the_registry_closes_what_it_loaded():
    registry, _, opened = _registry(['alpha', 'beta'])
    with registry.acquire('alpha'):
        pass
    registry.close()
    assert opened['alpha'][0].closed
    assert opened['beta'] == []       # never loaded, never closed
    registry.close()                  # idempotent: serve() and main() both call it


def test_two_models_cannot_share_a_name():
    with pytest.raises(ValueError, match='name'):
        server.ModelRegistry([
            server.ModelSpec(name='alpha', source='/a'),
            server.ModelSpec(name='alpha', source='/b'),
        ])


def _routed_server(names, **kwargs):
    registry, loads, opened = _registry(names, **kwargs)
    return server.UniRTHTTPServer(
        ('127.0.0.1', 0), None, names[0], registry=registry
    ), registry, loads, opened


def test_the_model_field_picks_which_model_answers():
    httpd, registry, _, _ = _routed_server(['alpha', 'beta'])
    try:
        status, body = _post_to(httpd, '/v1/chat/completions', {
            'model': 'beta', 'messages': [{'role': 'user', 'content': 'hi'}],
        })
        assert status == 200
        assert body['choices'][0]['message']['content'] == 'answered by beta'
        # The reply names the model that actually answered, not the default.
        assert body['model'] == 'beta'
    finally:
        registry.close()


def test_a_request_that_names_no_model_gets_the_first_one():
    httpd, registry, _, _ = _routed_server(['alpha', 'beta'])
    try:
        status, body = _post_to(httpd, '/v1/chat/completions', {
            'messages': [{'role': 'user', 'content': 'hi'}],
        })
        assert status == 200
        assert body['choices'][0]['message']['content'] == 'answered by alpha'
    finally:
        registry.close()


def test_asking_for_a_model_this_server_does_not_have_is_a_404():
    """The gap this closes: the field used to be read and dropped, so a client
    that switched models got the old one's answer with no way to tell."""
    httpd, registry, loads, _ = _routed_server(['alpha', 'beta'])
    try:
        status, body = _post_to(httpd, '/v1/chat/completions', {
            'model': 'gamma', 'messages': [{'role': 'user', 'content': 'hi'}],
        })
        assert status == 404
        assert 'alpha' in body['error']['message'] and 'beta' in body['error']['message']
        assert loads == {'alpha': 0, 'beta': 0}   # nothing was loaded to answer wrongly
    finally:
        registry.close()


def test_models_endpoint_lists_every_configured_model_loaded_or_not():
    httpd, registry, loads, _ = _routed_server(['alpha', 'beta'])
    import http.client

    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request('GET', '/v1/models')
            listed = json.loads(connection.getresponse().read())
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()
        registry.close()

    assert [entry['id'] for entry in listed['data']] == ['alpha', 'beta']
    # Listing must not load anything: the list is what can be served, not
    # what happens to be resident.
    assert loads == {'alpha': 0, 'beta': 0}


def test_legacy_completions_routes_on_the_model_field_too():
    httpd, registry, _, _ = _routed_server(['alpha', 'beta'])
    try:
        status, body = _post_to(httpd, '/v1/completions', {
            'model': 'beta', 'prompt': 'hi',
        })
        assert status == 200
        assert body['model'] == 'beta'
        assert body['choices'][0]['text'] == 'answered by beta'
    finally:
        registry.close()


def test_stats_reports_an_unloaded_model_as_unloaded_rather_than_404():
    """A model given back after its idle timeout is still served; a dashboard
    polling this must not read that as a missing endpoint."""
    httpd, registry, loads, _ = _routed_server(['alpha', 'beta'])
    import http.client

    port = httpd.server_address[1]
    worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request('GET', '/v1/stats')
            response = connection.getresponse()
            body = json.loads(response.read())
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        worker.join(timeout=5)
        httpd.server_close()
        registry.close()

    assert response.status == 200
    assert body['loaded_models'] == []
    assert loads == {'alpha': 0, 'beta': 0}


def test_one_model_still_answers_to_whatever_name_the_client_sends():
    """Compatibility, deliberately: with nothing to route to, rejecting the
    name would only break clients that hardcode an OpenAI one."""
    httpd, registry, _, _ = _routed_server(['alpha'])
    try:
        status, body = _post_to(httpd, '/v1/chat/completions', {
            'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'hi'}],
        })
        assert status == 200
        # Answered by the model that exists, and said so.
        assert body['model'] == 'alpha'
        assert body['choices'][0]['message']['content'] == 'answered by alpha'
    finally:
        registry.close()

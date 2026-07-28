# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import unirt.server as server

from unirt.server import (
    _parse_generation_args,
    _prepare_messages,
    _validate_messages,
    _with_serialized_model,
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
        return SimpleNamespace(text='ok', profile=profile)

    def runtime_stats(self):
        return {
            'model_bytes': 123,
            'kv_cache_bytes': 45,
            'device_peak_bytes': 678,
            'process_rss_bytes': 910,
            'device_name': 'Fake GPU',
        }


def test_successful_request_keeps_cached_state_for_the_next_one():
    """The cached prefix is the whole point: resetting after a good run would
    make every turn of a conversation re-prefill the entire transcript."""
    model = FakeModel()
    result = _with_serialized_model(model, threading.Lock(), lambda: model.generate('hi'))
    assert result.text == 'ok'
    assert model.reset_count == 0


def test_failed_generation_drops_cached_state():
    """A run that raised may leave the plugin's transcript out of step with its
    KV, and the next request would then reuse a prefix that is not really there."""
    model = FakeModel(fail=True)
    with pytest.raises(RuntimeError, match='boom'):
        _with_serialized_model(model, threading.Lock(), lambda: model.generate('hi'))
    assert model.reset_count == 1


def test_abandoned_stream_drops_cached_state():
    """A client that disconnects mid-stream aborts generation part-written."""
    model = FakeModel()

    def disconnect():
        raise BrokenPipeError('client went away')

    with pytest.raises(BrokenPipeError):
        _with_serialized_model(model, threading.Lock(), disconnect)
    assert model.reset_count == 1


def test_prefix_cache_can_be_turned_off():
    model = FakeModel()
    _with_serialized_model(
        model, threading.Lock(), lambda: model.generate('hi'), reuse_prefix=False
    )
    assert model.reset_count == 1


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
        served.append((current, model_id, host, port))

    monkeypatch.setattr(server, 'serve', fake_serve)
    monkeypatch.setattr(
        sys,
        'argv',
        ['unirt-server', '--model', 'acme/vision-GGUF', '--port', '9000'],
    )

    server.main()

    assert loaded == [('acme/vision-GGUF', {'device_map': 'llama_cpp', 'n_ctx': 0})]
    assert served == [(model, 'vision-GGUF', '127.0.0.1', 9000)]
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
    """Generation is serialized behind one lock; request_slots bounds the
    queue so a burst gets a fast 503 instead of piling up blocked threads."""
    import http.client

    httpd = server.UniRTHTTPServer(
        ('127.0.0.1', 0), FakeModel(), 'fake-model', max_queued_requests=1
    )
    httpd.request_slots.acquire()  # simulate the one slot already in flight
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

    def __init__(self, text: str, output):
        self._pieces = [text[i:i + 8] for i in range(0, len(text), 8)]
        self.output = output

    def __iter__(self):
        return iter(self._pieces)

    def cancel(self):
        self._pieces = []


class FakeToolModel(FakeModel):
    """Stands in for a grammar-constrained model that decided to call a tool."""

    def generate(self, _prompt, *, stream: bool = False, **_kwargs):
        profile = SimpleNamespace(stop_reason='eos', prompt_tokens=2, generated_tokens=9)
        out = SimpleNamespace(text=TOOL_CALL_JSON, profile=profile)
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

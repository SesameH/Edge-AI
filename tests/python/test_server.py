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
    _with_clean_model_state,
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


def test_request_state_is_reset_even_after_success():
    model = FakeModel()
    result = _with_clean_model_state(model, threading.Lock(), lambda: model.generate('hi'))
    assert result.text == 'ok'
    assert model.reset_count == 2


def test_request_state_is_reset_after_generation_error():
    model = FakeModel(fail=True)
    with pytest.raises(RuntimeError, match='boom'):
        _with_clean_model_state(model, threading.Lock(), lambda: model.generate('hi'))
    assert model.reset_count == 2


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
    monkeypatch.setattr(
        server,
        'serve',
        lambda current, model_id, host, port, max_queued_requests=8, embedding=None,
        embedding_id=None: served.append((current, model_id, host, port)),
    )
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

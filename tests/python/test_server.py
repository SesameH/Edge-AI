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
        lambda current, model_id, host, port, max_queued_requests=8: served.append(
            (current, model_id, host, port)
        ),
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

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from unirt.cli import _build_parser, _collect_media_history


def test_collect_media_history_keeps_all_turns():
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': '/one.png'},
                {'type': 'text', 'text': 'first'},
            ],
        },
        {'role': 'assistant', 'content': 'answer'},
        {
            'role': 'user',
            'content': [
                {'type': 'audio', 'audio': '/clip.wav'},
                {'type': 'image', 'image': '/two.jpg'},
            ],
        },
    ]

    assert _collect_media_history(messages) == (
        ['/one.png', '/two.jpg'],
        ['/clip.wav'],
    )


def test_embedding_cli_parses_multiple_texts():
    arguments = _build_parser().parse_args(
        ['embed', 'owner/model', 'first sentence', 'second sentence', '--device', 'cpu']
    )
    assert arguments.cmd == 'embed'
    assert arguments.texts == ['first sentence', 'second sentence']
    assert arguments.device == 'cpu'


def test_rerank_cli_parses_a_query_and_documents():
    arguments = _build_parser().parse_args(
        ['rerank', 'owner/reranker', 'a question', 'doc one', 'doc two', '--json']
    )
    assert arguments.cmd == 'rerank'
    assert arguments.query == 'a question'
    assert arguments.documents == ['doc one', 'doc two']
    assert arguments.json is True


def test_serve_cli_forwards_its_options_to_the_server(monkeypatch):
    """`unirt serve` is a front-end for unirt.server's own parser, so what
    matters is that every flag arrives there intact."""
    import unirt.server as server
    from unirt.cli import _cmd_serve

    seen = {}
    monkeypatch.setattr(server, 'main', lambda argv=None: seen.update(argv=argv))

    arguments = _build_parser().parse_args([
        'serve', 'owner/model', '--backend', 'mlx', '--port', '9999',
        '--rerank-model', 'owner/reranker', '--api-key', 's3cret',
        '--no-prefix-cache',
    ])
    _cmd_serve(arguments)

    argv = seen['argv']
    for flag, value in (
        ('--model', 'owner/model'),
        ('--backend', 'mlx'),
        ('--port', '9999'),
        ('--rerank-model', 'owner/reranker'),
        ('--api-key', 's3cret'),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert '--no-prefix-cache' in argv
    # Unset options must not be forwarded as the string "None".
    assert '--embedding-model' not in argv
    assert 'None' not in argv


def test_serve_cli_runs_without_a_chat_model():
    arguments = _build_parser().parse_args(['serve', '--embedding-model', 'owner/encoder'])
    assert arguments.model is None
    assert arguments.embedding_model == 'owner/encoder'

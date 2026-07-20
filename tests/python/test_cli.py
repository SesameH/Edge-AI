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

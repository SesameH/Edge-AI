# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Grammar-constrained decoding: json_schema / json_mode / raw GBNF.

The grammar physically masks tokens during decoding, so a schema-constrained
reply must parse as JSON and carry the required keys even from a 135M model.
"""

import json

import pytest

from unirt._ffi._api import UniRTError

SCHEMA = {
    'type': 'object',
    'properties': {
        'city': {'type': 'string'},
        'country': {'type': 'string'},
    },
    'required': ['city', 'country'],
}


class TestStructuredOutput:
    def test_json_schema_output_parses_and_validates(self, llama_model):
        out = llama_model.generate(
            'Give facts about the capital of France as JSON.',
            max_new_tokens=96,
            json_schema=SCHEMA,
        )
        data = json.loads(out.text)
        assert 'city' in data and 'country' in data

    def test_json_schema_accepts_serialized_string(self, llama_model):
        out = llama_model.generate(
            'Give facts about the capital of France as JSON.',
            max_new_tokens=96,
            json_schema=json.dumps(SCHEMA),
        )
        assert 'city' in json.loads(out.text)

    def test_json_mode_output_parses(self, llama_model):
        out = llama_model.generate(
            'Reply with a tiny JSON object greeting.',
            max_new_tokens=128,
            json_mode=True,
        )
        json.loads(out.text)

    def test_invalid_schema_rejected(self, llama_model):
        with pytest.raises(UniRTError):
            llama_model.generate('x', json_schema='{not json')

    def test_schema_and_json_mode_are_exclusive(self, llama_model):
        with pytest.raises(ValueError):
            llama_model.generate('x', json_mode=True, json_schema=SCHEMA)

    def test_schema_and_grammar_are_exclusive(self, llama_model):
        with pytest.raises(ValueError):
            llama_model.generate('x', grammar='root ::= "a"', json_schema=SCHEMA)

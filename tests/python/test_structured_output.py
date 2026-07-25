# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Grammar-constrained decoding: json_schema / json_mode / raw GBNF.

The grammar physically masks tokens during decoding, so a schema-constrained
reply must parse as JSON and carry the required keys even from a 135M model.
Both text backends are held to that, though they get there differently:
llama_cpp compiles the schema to GBNF, MLX runs its own pushdown automaton
(json_constraint.cpp). GBNF itself stays llama_cpp-only.
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
    def test_json_schema_output_parses_and_validates(self, constrained_model):
        out = constrained_model.generate(
            'Give facts about the capital of France as JSON.',
            max_new_tokens=96,
            json_schema=SCHEMA,
        )
        data = json.loads(out.text)
        assert 'city' in data and 'country' in data

    def test_json_schema_accepts_serialized_string(self, constrained_model):
        out = constrained_model.generate(
            'Give facts about the capital of France as JSON.',
            max_new_tokens=96,
            json_schema=json.dumps(SCHEMA),
        )
        assert 'city' in json.loads(out.text)

    def test_json_mode_output_parses(self, constrained_model):
        out = constrained_model.generate(
            'Reply with a tiny JSON object greeting.',
            max_new_tokens=128,
            json_mode=True,
        )
        json.loads(out.text)

    def test_enum_constrains_to_one_of_the_listed_values(self, constrained_model):
        out = constrained_model.generate(
            'Is Paris in France? Answer as JSON.',
            max_new_tokens=32,
            json_schema={
                'type': 'object',
                'properties': {'answer': {'enum': ['yes', 'no']}},
                'required': ['answer'],
            },
        )
        assert json.loads(out.text)['answer'] in ('yes', 'no')

    def test_nested_array_of_objects_is_well_formed(self, constrained_model):
        out = constrained_model.generate(
            'List two European cities as JSON.',
            max_new_tokens=128,
            json_schema={
                'type': 'object',
                'properties': {
                    'cities': {
                        'type': 'array',
                        'minItems': 2,
                        'maxItems': 2,
                        'items': {
                            'type': 'object',
                            'properties': {'name': {'type': 'string'}},
                            'required': ['name'],
                        },
                    },
                },
                'required': ['cities'],
            },
        )
        cities = json.loads(out.text)['cities']
        assert len(cities) == 2 and all(isinstance(city['name'], str) for city in cities)

    def test_invalid_schema_rejected(self, constrained_model):
        with pytest.raises(UniRTError):
            constrained_model.generate('x', json_schema='{not json')

    def test_schema_and_json_mode_are_exclusive(self, constrained_model):
        with pytest.raises(ValueError):
            constrained_model.generate('x', json_mode=True, json_schema=SCHEMA)

    def test_schema_and_grammar_are_exclusive(self, constrained_model):
        with pytest.raises(ValueError):
            constrained_model.generate('x', grammar='root ::= "a"', json_schema=SCHEMA)


class TestBackendSpecifics:
    def test_gbnf_grammar_runs_on_llama_cpp(self, llama_model):
        out = llama_model.generate(
            'Answer with a single word.', max_new_tokens=8, grammar='root ::= "yes" | "no"'
        )
        assert out.text in ('yes', 'no')

    def test_gbnf_grammar_is_refused_by_mlx(self, mlx_model):
        # llama.cpp's grammar dialect has no compiler on the MLX side; the
        # error has to be explicit rather than silently unconstrained output.
        with pytest.raises(UniRTError):
            mlx_model.generate('x', max_new_tokens=8, grammar='root ::= "yes" | "no"')

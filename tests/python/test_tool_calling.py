# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI tool calling: request validation, schema synthesis, and decoding.

The end-to-end tests matter most: they assert that the synthesized schema is
one the plugin can actually compile to a grammar, and that a 135M model
constrained by it emits a call naming a declared tool with arguments that
validate -- which is the whole point of spending the grammar slot rather than
asking the model politely.
"""

from __future__ import annotations

import json

import pytest

from unirt.tool_calling import (
    apply_tool_prompt,
    interpret_output,
    parse_tool_request,
    rewrite_tool_history,
)

WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Look up the current weather in a city.',
        'parameters': {
            'type': 'object',
            'properties': {'location': {'type': 'string'}},
            'required': ['location'],
            'additionalProperties': False,
        },
    },
}
CLOCK_TOOL = {
    'type': 'function',
    'function': {'name': 'get_time', 'description': 'Read the current time.'},
}


class TestRequestParsing:
    def test_no_tools_is_no_plan(self):
        assert parse_tool_request({}) is None

    def test_tool_choice_none_drops_the_tools(self):
        assert parse_tool_request({'tools': [WEATHER_TOOL], 'tool_choice': 'none'}) is None

    def test_auto_allows_a_text_reply(self):
        plan = parse_tool_request({'tools': [WEATHER_TOOL]})
        assert plan.must_call is False
        assert len(plan.schema()['oneOf']) == 2

    def test_required_removes_the_text_branch(self):
        plan = parse_tool_request({'tools': [WEATHER_TOOL], 'tool_choice': 'required'})
        # One tool, no text branch: the schema collapses to that tool alone.
        assert plan.schema()['properties']['name'] == {'const': 'get_weather'}

    def test_named_choice_selects_one_tool(self):
        plan = parse_tool_request({
            'tools': [WEATHER_TOOL, CLOCK_TOOL],
            'tool_choice': {'type': 'function', 'function': {'name': 'get_time'}},
        })
        assert plan.schema()['properties']['name'] == {'const': 'get_time'}

    def test_missing_parameters_become_an_empty_object(self):
        plan = parse_tool_request({'tools': [CLOCK_TOOL], 'tool_choice': 'required'})
        assert plan.schema()['properties']['arguments']['properties'] == {}

    @pytest.mark.parametrize(
        'payload',
        [
            {'tools': []},
            {'tools': 'get_weather'},
            {'tools': [{'type': 'retrieval'}]},
            {'tools': [{'type': 'function'}]},
            {'tools': [{'type': 'function', 'function': {'name': ''}}]},
            {'tools': [WEATHER_TOOL, WEATHER_TOOL]},
            {'tools': [WEATHER_TOOL], 'tool_choice': 'sometimes'},
            {'tools': [WEATHER_TOOL], 'tool_choice': {'type': 'function',
                                                      'function': {'name': 'nope'}}},
            {'tool_choice': 'required'},
        ],
    )
    def test_malformed_requests_are_client_errors(self, payload):
        with pytest.raises(ValueError):
            parse_tool_request(payload)


class TestHistoryRewriting:
    def test_assistant_call_is_replayed_as_the_json_it_produced(self):
        rewritten = rewrite_tool_history([{
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'get_weather', 'arguments': '{"location": "Taipei"}'},
            }],
        }])
        assert json.loads(rewritten[0]['content']) == {
            'name': 'get_weather',
            'arguments': {'location': 'Taipei'},
        }

    def test_tool_result_becomes_a_named_user_turn(self):
        rewritten = rewrite_tool_history([
            {
                'role': 'assistant',
                'tool_calls': [{
                    'id': 'call_1',
                    'function': {'name': 'get_weather', 'arguments': '{}'},
                }],
            },
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': '27C and clear'},
        ])
        assert rewritten[1] == {'role': 'user', 'content': 'Result from get_weather: 27C and clear'}

    def test_plain_turns_pass_through_untouched(self):
        messages = [{'role': 'user', 'content': 'hello'}]
        assert rewrite_tool_history(messages) == messages

    def test_unparseable_call_arguments_are_rejected(self):
        with pytest.raises(ValueError, match='not JSON'):
            rewrite_tool_history([{
                'role': 'assistant',
                'tool_calls': [{'function': {'name': 'get_weather', 'arguments': '{oops'}}],
            }])


class TestOutputInterpretation:
    def setup_method(self):
        self.plan = parse_tool_request({'tools': [WEATHER_TOOL]})

    def test_text_branch_unwraps_to_content(self):
        content, calls = interpret_output('{"content": "hi there"}', self.plan)
        assert (content, calls) == ('hi there', None)

    def test_call_branch_produces_openai_tool_calls(self):
        content, calls = interpret_output(
            '{"name": "get_weather", "arguments": {"location": "Taipei"}}', self.plan
        )
        assert content is None
        assert calls[0]['type'] == 'function'
        assert calls[0]['id'].startswith('call_')
        assert calls[0]['function'] == {
            'name': 'get_weather',
            'arguments': '{"location": "Taipei"}',
        }

    def test_truncated_output_stays_text_rather_than_a_partial_call(self):
        content, calls = interpret_output('{"name": "get_weat', self.plan)
        assert calls is None and content == '{"name": "get_weat'

    def test_undeclared_tool_name_is_not_promoted_to_a_call(self):
        content, calls = interpret_output('{"name": "rm_rf", "arguments": {}}', self.plan)
        assert calls is None and 'rm_rf' in content


class TestConstrainedGeneration:
    """The synthesized schema has to survive the plugin's grammar compiler."""

    def _generate(self, model, plan, user_text: str) -> str:
        messages = apply_tool_prompt([{'role': 'user', 'content': user_text}], plan)
        prompt = model._apply_chat_template(messages, True, False, None)
        out = model.generate(prompt, max_new_tokens=96, json_schema=plan.schema())
        model.reset()
        return out.text

    def test_required_choice_yields_a_valid_call(self, llama_model):
        plan = parse_tool_request({'tools': [WEATHER_TOOL], 'tool_choice': 'required'})
        content, calls = interpret_output(
            self._generate(llama_model, plan, 'What is the weather in Taipei?'), plan
        )
        assert content is None, f'expected a tool call, got text: {content!r}'
        assert calls[0]['function']['name'] == 'get_weather'
        assert isinstance(json.loads(calls[0]['function']['arguments'])['location'], str)

    def test_auto_choice_produces_one_of_the_two_shapes(self, llama_model):
        plan = parse_tool_request({'tools': [WEATHER_TOOL]})
        text = self._generate(llama_model, plan, 'Say hello.')
        payload = json.loads(text)
        assert ('content' in payload) ^ ('name' in payload)

    def test_named_choice_pins_the_tool(self, llama_model):
        plan = parse_tool_request({
            'tools': [WEATHER_TOOL, CLOCK_TOOL],
            'tool_choice': {'type': 'function', 'function': {'name': 'get_time'}},
        })
        _, calls = interpret_output(self._generate(llama_model, plan, 'What time is it?'), plan)
        assert calls[0]['function']['name'] == 'get_time'

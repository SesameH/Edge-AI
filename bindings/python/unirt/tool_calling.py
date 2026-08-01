# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI tool calling, built on grammar-constrained decoding.

The hosted API can let a model emit a proprietary tool-call token sequence. A
local 1B model cannot be trusted to invent one, so this module spends the
grammar slot instead: the tools a caller declares are compiled into a JSON
schema that the sampler physically cannot leave. A reply is therefore always
parseable, always names a declared tool, and always carries arguments that
validate against that tool's own parameter schema -- there is no "the model
returned malformed JSON" failure mode to retry around.

``tool_choice`` picks which branches the schema allows:

    none      -- tools are dropped entirely; a plain text turn
    auto      -- one text branch plus one branch per tool (the default)
    required  -- tool branches only, so the model must call something
    {"type": "function", ...} -- that one tool's branch alone

``parallel_tool_calls`` decides whether a turn may return more than one call.
It defaults to **false** here where the hosted API defaults to true, because
several calls at once means planning a whole batch before seeing any result,
which is what small models are worst at -- and a wrong second call is a tool
the caller actually executes. Clients that want it ask for it, and then the
grammar accepts ``{"tool_calls": [...]}`` instead of a bare call object.

The tool definitions are rendered into a system message here rather than passed
to the chat template, because ``llama_chat_apply_template`` has no tools
parameter (see ``llm.cpp``) and the MLX backend has no tool-aware template
either -- doing it in the prompt is the only thing that behaves identically on
both backends.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

__all__ = [
    'ToolPlan',
    'apply_tool_prompt',
    'interpret_output',
    'parse_tool_request',
    'rewrite_tool_history',
]

# Wrapping assistant text in JSON costs escaping and a little fluency, so the
# text branch is described to the model in the same breath as the tools.
_TEXT_BRANCH_KEY = 'content'


@dataclass(frozen=True)
class ToolPlan:
    """Everything a tool-enabled request needs, derived from the request body."""

    tools: tuple[dict, ...]
    forced_name: str | None
    must_call: bool
    parallel: bool = False

    def _call_branches(self) -> list[dict]:
        return [
            {
                'type': 'object',
                'properties': {
                    'name': {'const': tool['name']},
                    'arguments': tool['parameters'],
                },
                'required': ['name', 'arguments'],
                'additionalProperties': False,
            }
            for tool in self.tools
            if self.forced_name is None or tool['name'] == self.forced_name
        ]

    def schema(self) -> dict:
        """The JSON schema that constrains this turn's output."""
        branches = self._call_branches()
        if self.parallel:
            # minItems 1 so "called nothing" cannot be spelled as an empty
            # array; that is what the text branch is for.
            item = branches[0] if len(branches) == 1 else {'oneOf': branches}
            branches = [{
                'type': 'object',
                'properties': {
                    'tool_calls': {'type': 'array', 'items': item, 'minItems': 1},
                },
                'required': ['tool_calls'],
                'additionalProperties': False,
            }]
        if not self.must_call:
            branches.append({
                'type': 'object',
                'properties': {_TEXT_BRANCH_KEY: {'type': 'string'}},
                'required': [_TEXT_BRANCH_KEY],
                'additionalProperties': False,
            })
        return branches[0] if len(branches) == 1 else {'oneOf': branches}

    def system_prompt(self) -> str:
        """Instructions describing the tools and the reply format."""
        lines = ['You have access to these tools:', '']
        for tool in self.tools:
            if self.forced_name is not None and tool['name'] != self.forced_name:
                continue
            description = tool.get('description') or 'no description provided'
            lines.append(f'- {tool["name"]}: {description}')
            lines.append(f'  parameters: {json.dumps(tool["parameters"])}')
        lines.append('')
        if self.parallel:
            lines.append(
                'Reply with a single JSON object and nothing else. To call tools, use '
                '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}, ...]}. '
                'Only include a call whose result you need before you can answer.'
            )
        else:
            lines.append(
                'Reply with a single JSON object and nothing else. To call a tool, use '
                '{"name": "<tool name>", "arguments": {...}}.'
            )
        if not self.must_call:
            lines.append(
                f'To answer the user directly instead, use '
                f'{{"{_TEXT_BRANCH_KEY}": "<your reply>"}}.'
            )
        return '\n'.join(lines)


def _tool_name(function: dict) -> str:
    name = function.get('name')
    if not isinstance(name, str) or not name or '\x00' in name:
        raise ValueError('each tool needs a non-empty NUL-free function name')
    return name


def _tool_parameters(function: dict) -> dict:
    parameters = function.get('parameters')
    if parameters is None:
        # A no-argument tool still has to produce *something* the grammar can
        # accept, and `{}` is the JSON that means "no arguments".
        return {'type': 'object', 'properties': {}, 'additionalProperties': False}
    if not isinstance(parameters, dict):
        raise ValueError('tool function parameters must be a JSON Schema object')
    return parameters


def _parse_tool_choice(choice, names: set[str]) -> tuple[str | None, bool, bool]:
    """Return (forced_name, must_call, tools_enabled) for a tool_choice value."""
    if choice is None or choice == 'auto':
        return None, False, True
    if choice == 'none':
        return None, False, False
    if choice == 'required':
        return None, True, True
    if isinstance(choice, dict):
        function = choice.get('function')
        if choice.get('type') != 'function' or not isinstance(function, dict):
            raise ValueError(
                "tool_choice objects must be {'type': 'function', 'function': {'name': ...}}"
            )
        name = function.get('name')
        if name not in names:
            raise ValueError(f'tool_choice names an undeclared tool: {name!r}')
        return name, True, True
    raise ValueError("tool_choice must be 'auto', 'none', 'required', or a function object")


def parse_tool_request(req: dict) -> ToolPlan | None:
    """Validate ``tools``/``tool_choice``; None when this turn uses no tools."""
    tools = req.get('tools')
    if tools is None:
        if req.get('tool_choice') not in (None, 'none', 'auto'):
            raise ValueError('tool_choice was given without any tools')
        return None
    if not isinstance(tools, list) or not tools:
        raise ValueError('tools must be a non-empty array')

    normalized: list[dict] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError('each tool must be an object')
        if tool.get('type', 'function') != 'function':
            raise ValueError("only tools of type 'function' are supported")
        function = tool.get('function')
        if not isinstance(function, dict):
            raise ValueError("each tool needs a 'function' object")
        name = _tool_name(function)
        if name in names:
            raise ValueError(f'duplicate tool name: {name!r}')
        names.add(name)
        entry = {'name': name, 'parameters': _tool_parameters(function)}
        description = function.get('description')
        if description is not None:
            if not isinstance(description, str) or '\x00' in description:
                raise ValueError('tool description must be a NUL-free string')
            entry['description'] = description
        normalized.append(entry)

    forced_name, must_call, enabled = _parse_tool_choice(req.get('tool_choice'), names)
    if not enabled:
        return None
    parallel = req.get('parallel_tool_calls', False)
    if not isinstance(parallel, bool):
        raise ValueError('parallel_tool_calls must be a boolean')
    if parallel and forced_name is not None:
        # tool_choice named one tool; asking for several at once contradicts it.
        raise ValueError('parallel_tool_calls cannot be combined with a named tool_choice')
    return ToolPlan(tuple(normalized), forced_name, must_call, parallel)


def rewrite_tool_history(messages: list) -> list:
    """Flatten prior tool calls and results into plain text turns.

    Chat templates reached through ``llama_chat_apply_template`` only handle
    system/user/assistant roles, and an unknown role there is a hard template
    error rather than a degraded prompt. Prior assistant calls are re-rendered
    as the exact JSON the grammar had produced, so the transcript the model
    reads back matches what it was constrained to write; tool results become
    user turns, which is the only role every template accepts.
    """
    rewritten: list = []
    call_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            rewritten.append(message)
            continue
        role = message.get('role')
        if role == 'assistant' and message.get('tool_calls'):
            calls = message['tool_calls']
            if not isinstance(calls, list):
                raise ValueError('assistant tool_calls must be an array')
            rendered = []
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                    raise ValueError('each tool call needs a function object')
                function = call['function']
                name = _tool_name(function)
                arguments = function.get('arguments', '{}')
                if not isinstance(arguments, str):
                    raise ValueError('tool call arguments must be a JSON string')
                if isinstance(call.get('id'), str):
                    call_names[call['id']] = name
                try:
                    parsed = json.loads(arguments or '{}')
                except json.JSONDecodeError as exc:
                    raise ValueError(f'tool call arguments are not JSON: {exc}') from exc
                rendered.append(json.dumps({'name': name, 'arguments': parsed}))
            rewritten.append({'role': 'assistant', 'content': '\n'.join(rendered)})
            continue
        if role == 'tool':
            content = message.get('content')
            if not isinstance(content, str):
                raise ValueError('tool messages need string content')
            call_id = message.get('tool_call_id')
            name = message.get('name') or call_names.get(call_id if isinstance(call_id, str) else '')
            label = f'Result from {name}' if name else 'Tool result'
            rewritten.append({'role': 'user', 'content': f'{label}: {content}'})
            continue
        rewritten.append(message)
    return rewritten


def apply_tool_prompt(messages: list, plan: ToolPlan) -> list:
    """Prepend the tool instructions as a system turn."""
    return [{'role': 'system', 'content': plan.system_prompt()}, *messages]


def _tool_call(payload: dict, declared: set[str]) -> dict | None:
    """One call object in OpenAI's shape, or None if it does not name a tool."""
    name = payload.get('name')
    if not isinstance(name, str) or name not in declared:
        return None
    return {
        'id': 'call_' + uuid.uuid4().hex[:24],
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(payload.get('arguments', {}))},
    }


def interpret_output(text: str, plan: ToolPlan) -> tuple[str | None, list[dict] | None]:
    """Split constrained output into (content, tool_calls).

    The grammar guarantees the shape, so anything unparseable here means
    generation was cut short by max_tokens; that surfaces as plain text rather
    than a fabricated call, since a truncated call is not one the caller can
    safely execute.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, None
    if not isinstance(payload, dict):
        return text, None
    if _TEXT_BRANCH_KEY in payload and 'name' not in payload:
        content = payload[_TEXT_BRANCH_KEY]
        return (content if isinstance(content, str) else text), None

    declared = {tool['name'] for tool in plan.tools}
    batch = payload.get('tool_calls')
    if isinstance(batch, list):
        calls = [
            call
            for item in batch
            if isinstance(item, dict) and (call := _tool_call(item, declared)) is not None
        ]
        # An empty batch is not "no tools" -- the grammar has a text branch for
        # that -- it is output that got cut off, so surface it as text.
        return (None, calls) if calls else (text, None)

    call = _tool_call(payload, declared)
    return (None, [call]) if call is not None else (text, None)

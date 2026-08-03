# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Several handles on one context, decoding in one batch.

`n_seq_max` on the llama.cpp backend means the handles opened with it may share
a context and travel through llama_decode together. That is where a slot pool's
throughput comes from -- four sequences decoded separately are four passes over
the weights, and batched they are one -- and it is also the change most able to
break things quietly, because every handle now writes into KV cells its
neighbours can see.

So the assertions here are about isolation and contract, not about tokens. A
shared pool changes the arithmetic: the cells another sequence occupies are
masked out of the attention, but they are still part of the tensor being
reduced, and a near-tie between two tokens can land the other way. The same
request run alone and run beside three others is allowed to differ by a token
in the middle of a sentence -- and *not* allowed to answer somebody else's
question, lose its own prefix, or ignore its own context limit.
"""

from __future__ import annotations

import threading

import pytest
from conftest import model_path

from unirt._ffi._api import UniRTError
from unirt.auto import AutoModelForCausalLM

SLOTS = 4

# Distinct, unambiguous, and short enough for a 135M model to get right. If a
# sequence read another one's cells this is what would break.
QUESTIONS = [
    ('What is the capital of France? Answer with one word.', 'paris'),
    ('What colour is grass? Answer with one word.', 'green'),
    ('How many days are in a week? Answer with a number.', '7'),
    ('What language is spoken in Japan? Answer with one word.', 'japan'),
]


@pytest.fixture
def pool(sdk):
    """SLOTS handles that share one batching context."""
    path = model_path('gguf')
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, device_map='llama_cpp', n_ctx=512, n_seq_max=SLOTS)
        for _ in range(SLOTS)
    ]
    yield models
    for model in models:
        model.close()


@pytest.fixture
def lone(sdk):
    """One handle with a context to itself, for comparison."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path('gguf'), device_map='llama_cpp', n_ctx=512)
    yield model
    model.close()


def ask(model, question, **kwargs):
    model.reset()
    prompt = model._apply_chat_template(
        [{'role': 'user', 'content': question}], True, False, None)
    kwargs.setdefault('max_new_tokens', 24)
    kwargs.setdefault('temperature', 0.0)
    return model.generate(prompt, **kwargs)


def concurrently(work, count):
    """Run work(i) for each i at the same time; re-raise the first failure."""
    results: list = [None] * count
    failures: list = []

    def run(index):
        try:
            results[index] = work(index)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failures.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]
    return results


# ---------------------------------------------------------------------------
# Isolation: the thing a shared context could take away
# ---------------------------------------------------------------------------


def test_each_sequence_answers_its_own_question(pool):
    """The load-bearing test. Four different prompts decoded in one batch, and
    every answer has to belong to the prompt that asked for it -- a sequence id
    off by one, or an attention mask that let a neighbour's cells through,
    shows up here and almost nowhere else."""
    answers = concurrently(lambda index: ask(pool[index], QUESTIONS[index][0]).text, SLOTS)
    for (question, expected), answer in zip(QUESTIONS, answers):
        assert expected in answer.lower(), f'{question!r} was answered {answer!r}'


def test_a_batched_answer_matches_the_same_handle_alone(pool):
    """Not token equality -- a shared pool is allowed to move a near-tie. What
    a sequence may not do is come back with a different answer."""
    batched = concurrently(lambda index: ask(pool[index], QUESTIONS[index][0]).text, SLOTS)
    for index, (question, expected) in enumerate(QUESTIONS):
        alone = ask(pool[index], question).text
        assert expected in alone.lower()
        assert expected in batched[index].lower()


def test_resetting_one_sequence_leaves_the_others_holding_their_cache(pool):
    """reset() used to clear the whole context. With four conversations on one
    context that would silently throw away three of them."""
    first = ask(pool[0], QUESTIONS[0][0])
    ask(pool[1], QUESTIONS[1][0])
    assert pool[1].runtime_stats()['kv_cache_bytes'] > 0

    pool[0].reset()
    # The neighbour's cache survived, so re-asking its own question reuses it.
    again = ask(pool[1], QUESTIONS[1][0])
    assert QUESTIONS[1][1] in again.text.lower()
    # And the one that was reset starts from nothing and still works.
    assert QUESTIONS[0][1] in ask(pool[0], QUESTIONS[0][0]).text.lower()
    assert first.profile.generated_tokens > 0


def test_one_sequence_filling_up_does_not_starve_another(sdk):
    """The pool is sized for every sequence at its full window, so a sequence
    that runs to its own context limit reports overflow for itself and takes
    nothing away from its neighbour."""
    path = model_path('gguf')
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, device_map='llama_cpp', n_ctx=64, n_seq_max=2)
        for _ in range(2)
    ]
    try:
        long_prompt = models[0]._apply_chat_template(
            [{'role': 'user', 'content': 'Count from one to one hundred, one per line.'}],
            True, False, None)
        overflowed = models[0].generate(long_prompt, max_new_tokens=600, temperature=0.0)
        assert overflowed.profile.stop_reason == 'context_length'
        # The neighbour never asked for anything unusual and must be unaffected.
        assert QUESTIONS[0][1] in ask(models[1], QUESTIONS[0][0]).text.lower()
    finally:
        for model in models:
            model.close()


def test_the_prefix_cache_is_per_sequence(pool):
    """Two handles, same prompt, one context. The second must reuse its own
    cached prefix and not the first's -- and either way must answer."""
    ask(pool[0], QUESTIONS[0][0])
    ask(pool[1], QUESTIONS[0][0])
    # A second turn on each: the prefix each one reuses is its own.
    for index in (0, 1):
        model = pool[index]
        prompt = model._apply_chat_template(
            [{'role': 'user', 'content': QUESTIONS[0][0]}], True, False, None)
        out = model.generate(prompt, max_new_tokens=24, temperature=0.0)
        assert QUESTIONS[0][1] in out.text.lower()


# ---------------------------------------------------------------------------
# The contract, unchanged by sharing
# ---------------------------------------------------------------------------


def test_streaming_still_equals_blocking_under_load(pool):
    """Every handle streams while three others decode into the same batch."""
    def one(index):
        blocking = ask(pool[index], QUESTIONS[index][0], max_new_tokens=16).text
        streamed = ''.join(
            ask(pool[index], QUESTIONS[index][0], max_new_tokens=16, stream=True))
        return blocking, streamed

    for blocking, streamed in concurrently(one, SLOTS):
        assert streamed == blocking


def test_stop_sequences_hold_under_load(pool):
    def one(index):
        text = ask(pool[index], QUESTIONS[index][0], max_new_tokens=24).text
        assert len(text) > 4
        stop = text[2:5]
        out = ask(pool[index], QUESTIONS[index][0], max_new_tokens=24, stop=[stop])
        return stop, out

    for stop, out in concurrently(one, SLOTS):
        assert out.profile.stop_reason == 'stop_sequence'
        assert stop not in out.text


def test_a_rejected_request_does_not_disturb_the_batch(pool):
    """One sequence sends something invalid while the others are mid-flight.
    Its own handle has to survive it, and theirs must not notice."""
    def one(index):
        if index == 0:
            with pytest.raises(UniRTError):
                ask(pool[0], QUESTIONS[0][0], n_past=2_000_000_000)
            return ask(pool[0], QUESTIONS[0][0]).text
        return ask(pool[index], QUESTIONS[index][0]).text

    answers = concurrently(one, SLOTS)
    for (_, expected), answer in zip(QUESTIONS, answers):
        assert expected in answer.lower()


def test_kv_cache_bytes_are_this_sequence_s_share(pool):
    """Not the whole pool: four handles reporting the pool size would read as
    four times the memory actually in use."""
    ask(pool[0], QUESTIONS[0][0], max_new_tokens=4)
    small = pool[0].runtime_stats()['kv_cache_bytes']
    ask(pool[0], QUESTIONS[0][0], max_new_tokens=40)
    assert pool[0].runtime_stats()['kv_cache_bytes'] > small
    # A sequence that has decoded nothing is not paying for its neighbours.
    assert pool[3].runtime_stats()['kv_cache_bytes'] < small


def test_saving_and_loading_a_session_moves_one_sequence_only(pool, tmp_path):
    path = str(tmp_path / 'slot.kv')
    ask(pool[0], QUESTIONS[0][0])
    pool[0].save_kv_cache(path)
    ask(pool[1], QUESTIONS[1][0])

    pool[0].reset()
    pool[0].load_kv_cache(path)
    assert QUESTIONS[0][1] in ask(pool[0], QUESTIONS[0][0]).text.lower()
    # Restoring one sequence must not have written over another's cells.
    assert QUESTIONS[1][1] in ask(pool[1], QUESTIONS[1][0]).text.lower()


# ---------------------------------------------------------------------------
# Borrowing a sibling's prefix instead of evaluating it again
# ---------------------------------------------------------------------------

# Long enough that evaluating it is the whole cost of the request, which is
# what makes the timing assertion below safe by a wide margin.
SYSTEM = 'You are terse. ' + ' '.join(
    f'Background fact {index}: the value is {index * 7 % 97}.' for index in range(140))


def ask_with_system(model, question, **kwargs):
    prompt = model._apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}],
        True, False, None)
    kwargs.setdefault('max_new_tokens', 12)
    kwargs.setdefault('temperature', 0.0)
    return model.generate(prompt, **kwargs)


@pytest.fixture
def wide_pool(sdk):
    """Slots with room for a long shared system prompt."""
    path = model_path('gguf')
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, device_map='llama_cpp', n_ctx=4096, n_seq_max=SLOTS)
        for _ in range(SLOTS)
    ]
    yield models
    for model in models:
        model.close()


def test_an_idle_sibling_s_prefix_is_taken_rather_than_re_evaluated(wide_pool):
    """The ordinary shape of serving: several clients, one system prompt. With
    a unified cache the cells are shared rather than copied, so the second
    client's prompt costs nothing to put in place.

    Timed, because there is no other outward sign that it happened -- but the
    measured gap is ~50x on a 1854-token prompt, so a 3x threshold is not a
    close call on any machine.
    """
    import time

    started = time.monotonic()
    first = ask_with_system(wide_pool[0], QUESTIONS[0][0])
    cold = time.monotonic() - started
    assert QUESTIONS[0][1] in first.text.lower()

    started = time.monotonic()
    second = ask_with_system(wide_pool[1], QUESTIONS[1][0])
    warm = time.monotonic() - started
    assert QUESTIONS[1][1] in second.text.lower()
    assert warm * 3 < cold, f'borrowing saved nothing: {cold:.2f}s then {warm:.2f}s'


def test_a_borrowed_prefix_brings_none_of_the_donor_s_own_turn(wide_pool):
    """Only the shared leading tokens move. Everything the donor decoded after
    them is its own, and must not appear in the borrower's context."""
    ask_with_system(wide_pool[0], QUESTIONS[0][0])
    for index in (1, 2, 3):
        answer = ask_with_system(wide_pool[index], QUESTIONS[index][0]).text.lower()
        assert QUESTIONS[index][1] in answer
        # The donor asked about France; a borrower that inherited its answer
        # too would be answering that instead of its own question.
        assert QUESTIONS[0][1] not in answer


def test_shifting_never_starts_inside_a_shared_region(sdk):
    """A sequence that borrowed shares real cells with the one it borrowed
    from, and llama.cpp renumbers a cell for every sequence holding it
    (llama_kv_cache::seq_add shifts any cell whose seq set contains the id).
    Evicting inside a borrowed prefix would therefore renumber the donor's
    conversation while its own transcript still says otherwise.

    Asserted on the eviction decision rather than on the reply, because the
    damage is soft: the tokens stay, only their positions move, and a model
    answering from a corrupted context still mostly answers. Both obvious
    behavioural tests -- the donor re-asking, the borrower recalling something
    only the prefix knows -- pass with the guard deliberately removed, the
    borrower because eviction shortens its transcript too so its next turn
    simply re-prefills. What does not survive removal is this: the eviction
    log reports head=4 instead of head=<the borrowed length>.
    """
    import logging
    import re

    from unirt._ffi import _api

    path = model_path('gguf')
    shared = 'You are terse. ' + ' '.join(
        f'Background fact {index}: the value is {index}.' for index in range(30))

    def ask(model, question, **kwargs):
        prompt = model._apply_chat_template(
            [{'role': 'system', 'content': shared}, {'role': 'user', 'content': question}],
            True, False, None)
        kwargs.setdefault('max_new_tokens', 12)
        kwargs.setdefault('temperature', 0.0)
        return model.generate(prompt, **kwargs)

    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Collect()
    logger = logging.getLogger('unirt')
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    _api.set_log_level('debug')
    _api.install_log_callback()

    models = [
        AutoModelForCausalLM.from_pretrained(
            path, device_map='llama_cpp', n_ctx=512, n_seq_max=2)
        for _ in range(2)
    ]
    try:
        assert QUESTIONS[0][1] in ask(models[0], QUESTIONS[0][0]).text.lower()
        borrower = ask(
            models[1], 'Count from one to one hundred, one number per line.',
            max_new_tokens=400, sliding_window=True)
        assert borrower.text
        assert borrower.profile.stop_reason != 'context_length', 'nothing was evicted'
    finally:
        for model in models:
            model.close()
        logger.removeHandler(handler)
        logger.setLevel(previous)

    borrowed = [int(m) for m in re.findall(r'borrowed (\d+) cached tokens', ' '.join(lines))]
    assert borrowed, 'the second sequence never borrowed a prefix'
    heads = [int(m) for m in re.findall(r'context shift evicted \d+ tokens \(head (\d+)', ' '.join(lines))]
    assert heads, 'the borrower never had to shift its window'
    assert min(heads) >= max(borrowed), (
        f'eviction started at {min(heads)}, inside a prefix of {max(borrowed)} shared cells')


def test_a_draft_model_keeps_its_own_context(sdk):
    """Speculative decoding verifies several positions of a batch it owns, so
    it cannot share. Asking for both is not an error -- the handle just gets a
    context to itself and says so."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path('gguf'), device_map='llama_cpp', n_ctx=512, n_seq_max=SLOTS,
        draft_model=model_path('gguf'))
    try:
        assert QUESTIONS[0][1] in ask(model, QUESTIONS[0][0]).text.lower()
    finally:
        model.close()

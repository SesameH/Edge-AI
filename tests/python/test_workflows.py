# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Lint the GitHub Actions workflows.

A workflow file is only ever exercised by pushing it, and GitHub rejects the
whole file on a structural error -- every job in it is skipped and the run
fails with no logs, which looks exactly like a runner or billing problem. These
checks catch the classes of mistake that have actually happened here.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip('yaml')

WORKFLOWS = sorted(
    (pathlib.Path(__file__).resolve().parents[2] / '.github' / 'workflows').glob('*.yml')
)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys.

    PyYAML keeps the last of a repeated key and says nothing, so a workflow
    with two jobs of the same name parses cleanly here while GitHub rejects it
    with "'<name>' is already defined". That divergence is what let a duplicate
    `windows:` job reach main.
    """


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f'duplicate key {key!r}', key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _load(path: pathlib.Path) -> dict:
    return yaml.load(path.read_text(encoding='utf-8'), Loader=_StrictLoader)


def test_there_are_workflows_to_check():
    assert WORKFLOWS, 'no workflow files found — has the path moved?'


@pytest.mark.parametrize('path', WORKFLOWS, ids=lambda p: p.name)
def test_workflow_has_no_duplicate_keys(path):
    _load(path)


@pytest.mark.parametrize('path', WORKFLOWS, ids=lambda p: p.name)
def test_every_job_has_steps_and_a_runner(path):
    # `on` is the YAML 1.1 boolean True, not the string, once parsed.
    workflow = _load(path)
    jobs = workflow['jobs']
    assert jobs, f'{path.name} defines no jobs'
    for name, job in jobs.items():
        assert 'runs-on' in job, f'{path.name}: job {name} has no runs-on'
        assert job.get('steps'), f'{path.name}: job {name} has no steps'


@pytest.mark.parametrize('path', WORKFLOWS, ids=lambda p: p.name)
def test_job_dependencies_exist(path):
    """A `needs:` naming a job that no longer exists is another whole-file error."""
    jobs = _load(path)['jobs']
    for name, job in jobs.items():
        needs = job.get('needs', [])
        for required in [needs] if isinstance(needs, str) else needs:
            assert required in jobs, f'{path.name}: job {name} needs missing job {required}'

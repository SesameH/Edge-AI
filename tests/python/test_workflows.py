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


PUBLISHING_MARKERS = ('gh-action-pypi-publish', 'gh release create', 'git push')


@pytest.mark.parametrize('path', WORKFLOWS, ids=lambda p: p.name)
def test_a_hand_dispatched_run_cannot_publish(path):
    """A workflow that can be dispatched by hand must not publish when it is.

    publish-sdk.yml exists to be triggered by a tag, but its build wiring --
    Vulkan, the CPU variants, manylinux -- is then only ever exercised while
    cutting a release, which is a bad moment to discover a broken flag. It
    takes workflow_dispatch so it can be dry-run. That is only a dry run if
    every job that leaves the repository stays behind a tag push.
    """
    import json

    workflow = _load(path)
    triggers = workflow[True]
    if 'workflow_dispatch' not in triggers:
        pytest.skip(f'{path.name} cannot be dispatched by hand')

    for name, job in workflow['jobs'].items():
        body = json.dumps(job)
        marker = next((m for m in PUBLISHING_MARKERS if m in body), None)
        if marker is None:
            continue
        condition = job.get('if', '')
        assert "github.event_name == 'push'" in condition, (
            f"{path.name}: job {name} runs {marker!r} but is not gated on a tag "
            f"push, so a hand-dispatched dry run would publish (if: {condition!r})"
        )


def test_the_dry_run_version_is_one_the_wheel_builder_accepts():
    """The dry run failed on this: it invented 0.0.0.dev0, and build_wheel.py
    takes X.Y.Z only, so every wheel job died at packaging having already
    built its native libraries. A dry run that cannot reach the packaging step
    does not test the packaging step."""
    import re
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'bindings' / 'python'))
    import build_wheel

    workflow = (root / '.github' / 'workflows' / 'publish-sdk.yml').read_text(encoding='utf-8')
    fallbacks = re.findall(r'echo "number=([^"]+)" >> "\$GITHUB_OUTPUT"', workflow)
    literals = [value for value in fallbacks if '$' not in value]
    assert literals, 'the version job no longer has a literal fallback version'
    for value in literals:
        assert build_wheel._VERSION_RE.fullmatch(value), (
            f'the workflow builds version {value!r}, which build_wheel.py rejects'
        )

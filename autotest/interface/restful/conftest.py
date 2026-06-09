"""Shared fixtures for config-driven RESTful interface tests."""

from __future__ import annotations

import pytest
from utils.config_utils import get_workerid
from utils.constant import DEFAULT_PORT, DEFAULT_SERVER
from utils.interface_utils import should_skip_marked_test
from utils.run_restful_chat import start_openai_service, terminate_restful_api

_INTERFACE_MODULES = (
    'autotest.interface.restful.test_restful_chat_completions_v1',
    'autotest.interface.restful.test_restful_completions_v1',
    'autotest.interface.restful.test_restful_generate',
    'autotest.interface.restful.reasoning_parser.test_reasoning_api',
    'autotest.interface.restful.tool_parser.test_tool_call_basic',
    'autotest.interface.restful.tool_parser.test_tool_call_advanced',
)

_SKIP_MARKERS = (
    'not_pytorch',
    'not_turbomind',
    'generate_logprob',
    'generate_experts',
)


def _run_config_from_item(item: pytest.Item) -> dict | None:
    callspec = getattr(item, 'callspec', None)
    if callspec is None:
        return None
    return callspec.params.get('run_config')


def pytest_collection_modifyitems(config, items):
    for item in items:
        run_config = _run_config_from_item(item)
        if not isinstance(run_config, dict):
            continue
        for marker_name in _SKIP_MARKERS:
            if marker_name not in {m.name for m in item.iter_markers()}:
                continue
            reason = should_skip_marked_test(marker_name, run_config)
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope='class')
def backend(run_config):
    return run_config['backend']


@pytest.fixture(scope='class')
def model_case(run_config):
    return run_config['model']


@pytest.fixture(scope='class')
def model_name(run_config):
    return run_config['model']


@pytest.fixture(scope='class')
def restful_server(request, config, worker_id):
    """Start one API server per parametrized ``run_config`` (class scope)."""
    run_config = request.getfixturevalue('run_config')
    pid, err = start_openai_service(config, run_config, worker_id, timeout=240)
    port = DEFAULT_PORT + get_workerid(worker_id)
    base_url = f'http://{DEFAULT_SERVER}:{port}'
    assert pid > 0, f'Failed to start RESTful API server: {err}'
    try:
        yield base_url
    finally:
        terminate_restful_api(worker_id)


@pytest.fixture(scope='class', autouse=True)
def _bind_restful_base_url(request, restful_server, monkeypatch):
    """Point module-level ``BASE_URL`` at the class-scoped server."""
    if request.cls is None:
        return
    module = request.module
    if module.__name__ not in _INTERFACE_MODULES:
        return
    monkeypatch.setattr(module, 'BASE_URL', restful_server, raising=False)
    request.cls.BASE_URL = restful_server
    import utils.tool_reasoning_definitions as trd
    monkeypatch.setattr(trd, 'BASE_URL', restful_server, raising=False)

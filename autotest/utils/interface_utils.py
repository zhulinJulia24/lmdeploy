"""Helpers for ``test_coverage: interface`` RESTful API tests."""

from __future__ import annotations

from typing import Any

from utils.config_utils import get_case_str_by_config


def interface_config_id(run_config: dict[str, Any]) -> str:
    return get_case_str_by_config(run_config)


def interface_suites(run_config: dict[str, Any]) -> list[str]:
    iface = run_config.get('interface')
    if isinstance(iface, list):
        return list(iface)
    if isinstance(iface, dict):
        return list(iface.get('suites') or [])
    return []


def parametrize_interface(suite: str):
    """Class decorator: ``@parametrize_interface('chat_completions_v1')``."""
    import pytest
    from utils.config_utils import get_interface_config_list

    def decorate(cls):
        configs = get_interface_config_list(suite)
        if not configs:
            return pytest.mark.skip(reason=f'no interface configs for suite={suite!r}')(cls)
        return pytest.mark.parametrize('run_config', configs, ids=interface_config_id)(cls)

    return decorate


def should_skip_marked_test(marker_name: str, run_config: dict[str, Any]) -> str | None:
    """Return skip reason when a pytest marker does not apply to
    *run_config*."""
    if marker_name in ('not_pytorch', 'not_turbomind'):
        backend = run_config.get('backend')
        blocked = marker_name.replace('not_', '')
        if backend == blocked:
            return f'not for {backend}'
    suites = set(interface_suites(run_config))
    if marker_name == 'generate_logprob':
        if 'generate_logprob' not in suites:
            return 'interface missing generate_logprob'
    if marker_name == 'generate_experts':
        if 'generate_experts' not in suites:
            return 'interface missing generate_experts'
    return None

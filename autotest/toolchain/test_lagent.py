import os

import pytest
from utils.config_utils import (
    TEST_COVERAGE_KEY,
    get_config,
    get_deps_profile_selector,
    get_model_path_from_config,
    _iter_per_model_entries,
    _model_matrix_env_key,
)

_IM_END = '<|im_end|>\n'

# Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 instruct ChatML (HF apply_chat_template).
QWEN3_CHATML_META = [
    dict(
        role='system',
        begin=dict(
            with_name='<|im_start|>system\n',
            without_name='<|im_start|>system\n',
        ),
        end=_IM_END,
    ),
    dict(
        role='user',
        begin=dict(
            with_name='<|im_start|>user\n',
            without_name='<|im_start|>user\n',
        ),
        end=_IM_END,
    ),
    dict(
        role='assistant',
        begin=dict(
            with_name='<|im_start|>assistant\n',
            without_name='<|im_start|>assistant\n',
        ),
        end=_IM_END,
    ),
]

QWEN3_CHATML_STOP_WORDS = ['<|im_end|>']


def _lagent_meta_for_model(model_id: str) -> list[dict]:
    if 'qwen' in (model_id or '').lower():
        return QWEN3_CHATML_META
    from lagent.llms import INTERNLM2_META

    return INTERNLM2_META


def _lagent_stop_words_for_model(model_id: str) -> list[str]:
    if 'qwen' in (model_id or '').lower():
        return list(QWEN3_CHATML_STOP_WORDS)
    return list(QWEN3_CHATML_STOP_WORDS)


def _get_lagent_model_list() -> list[str]:
    config = get_config()
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    models: list[str] = []
    for model_id, entry in _iter_per_model_entries(env_key, deps_profile):
        if 'lagent' not in (entry.get(TEST_COVERAGE_KEY) or []):
            continue
        if model_id not in models:
            models.append(model_id)
    return models


def _resolve_lagent_pipeline_path(config, model_id: str) -> str:
    if config.get('model_path_layout') == 'hf_hub':
        return model_id
    if '/' in model_id and not os.path.isabs(model_id):
        return model_id
    return get_model_path_from_config(config, model_id)


@pytest.mark.order(10)
@pytest.mark.lagent
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('model', _get_lagent_model_list())
def test_repeat(config, model):
    from lagent.llms import LMDeployPipeline

    llm = LMDeployPipeline(
        path=_resolve_lagent_pipeline_path(config, model),
        meta_template=_lagent_meta_for_model(model),
        tp=1,
        top_k=40,
        top_p=0.8,
        temperature=1.2,
        stop_words=_lagent_stop_words_for_model(model),
        max_new_tokens=4096,
    )
    user_msg = [{
        'role':
        'user',
        'content':
        '已知$$z_{1}=1$$,$$z_{2}=\\text{i}$$,$$z_{3}=-1$$,$$z_{4}=-\\text{i}$$,顺次连结它们所表示的点,则所得图形围成的面积为（ ）\nA. $$\\dfrac{1}{4}$$\n B. $$\\dfrac{1}{2}$$\n C. $$1$$\n D. $$2$$\n\n'  # noqa: E501
    }]
    response_list = []
    for i in range(3):
        print(f'run_{i}：')
        prompt = llm.template_parser(user_msg)
        response = llm.generate(prompt, do_preprocess=False)
        print(response)
        response_list.append(response)
        assert len(response) > 10
    assert response_list[0] != response_list[1] and response_list[1] != response_list[2]

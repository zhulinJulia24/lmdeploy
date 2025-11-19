from functools import partial

from vlmeval.api import *  # noqa 821

api_models = {
    # lmdeploy api
    'lmdeploy_port23333':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'lmdeploy_port23334':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=100,
    ),
    'lmdeploy_port23335':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23336':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23337':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23338':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23339':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23340':
    partial(
        LMDeployAPI,  # noqa 821
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
}

supported_VLM = {}

model_groups = [api_models]

for grp in model_groups:
    supported_VLM.update(grp)

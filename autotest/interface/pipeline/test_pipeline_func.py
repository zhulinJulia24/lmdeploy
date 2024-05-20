import os
from multiprocessing import Process

import pytest
from utils.config_utils import get_cuda_id_by_workerid
from utils.pipeline_chat import (assert_pipeline_common_log,
                                 assert_pipeline_single_return,
                                 save_pipeline_common_log)
from utils.restful_return_check import get_repeat_times

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline


@pytest.mark.order(8)
@pytest.mark.pipeline_turbomind_func
@pytest.mark.timeout(240)
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
class TestPipelineTurbomindFuncRegression:

    def test_return_with_prompt(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            response = pipe('Hi, pls intro yourself')

            result = assert_pipeline_single_return(response)
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_backend_config_tp(self, config, model):
        with pytest.raises(AssertionError, match='tp should be 2\\^n'):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=100)
            pipe = pipeline(model_path, backend_config=backend_config)
            del pipe

    def test_backend_config_session_len(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(session_len=10, tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'])

            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length'
                result &= response[i].generate_token_len == 0
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_min_new_tokens(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test min_new_tokens
            gen_config = GenerationConfig(min_new_tokens=200, ignore_eos=True)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length'
                result &= response[i].session_id == i
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_stop_words(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test stop_words
            gen_config = GenerationConfig(stop_words=[' and', '浦', ' to'],
                                          random_seed=1,
                                          temperature=0.01)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= '浦' not in response[i].text
                result &= ' and' not in response[
                    i].text and ' to ' not in response[i].text
                result &= response[i].finish_reason == 'stop' and response[
                    i].generate_token_len < 20
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_bad_words(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test bad_words
            gen_config = GenerationConfig(bad_words=[' and', '浦', ' to'],
                                          temperature=0.01,
                                          random_seed=1)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= '浦' not in response[i].text and '蒲' in response[
                    i].text
                result &= ' and' not in response[
                    i].text and ' to ' not in response[i].text
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_special_words(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test special_words
            gen_config = GenerationConfig(skip_special_tokens=False)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length' or response[
                    i].finish_reason == 'stop'
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_max_new_tokens(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test max_new_tokens
            gen_config = GenerationConfig(max_new_tokens=5)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length'
                result &= response[i].generate_token_len == 6
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_ignore_eos(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test max_new_tokens with ignore_eos
            gen_config = GenerationConfig(ignore_eos=True, max_new_tokens=256)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length'
                result &= response[i].generate_token_len == 257
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    def test_gen_config_repetition_penalty(self, config, model, worker_id):

        def run_pipeline_testcase(config, model, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = TurbomindEngineConfig(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test repetition_penalty
            gen_config = GenerationConfig(repetition_penalty=0.01,
                                          random_seed=1)
            response = pipe('Shanghai is', gen_config=gen_config)

            result = 'a 上海 is a 上海, ' * 10 in response.text \
                or get_repeat_times(response.text, 'Shanghai is') > 5
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

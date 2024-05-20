import os
from multiprocessing import Process

import pytest
from pytest import assume
from utils.config_utils import get_cuda_id_by_workerid
from utils.pipeline_chat import (assert_pipeline_common_log,
                                 save_pipeline_common_log)
from utils.restful_return_check import get_repeat_times

from lmdeploy import (GenerationConfig, PytorchEngineConfig,
                      TurbomindEngineConfig, pipeline)


@pytest.mark.order(8)
@pytest.mark.pipeline_turbomind_func
@pytest.mark.timeout(240)
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
class TestPipelineTurbomindFuncRegression:

    @pytest.mark.parametrize('backend', [TurbomindEngineConfig])
    def test_backend_config_tp(self, config, model, backend):
        with pytest.raises(AssertionError, match='tp should be 2\\^n'):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=100)
            pipe = pipeline(model_path, backend_config=backend_config)
            del pipe

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_backend_config_session_len(self, config, model, backend,
                                        worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(session_len=10, tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_min_new_tokens(self, config, model, backend,
                                       worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
            pipe = pipeline(model_path, backend_config=backend_config)
            # test min_new_tokens
            gen_config = GenerationConfig(min_new_tokens=200, ignore_eos=True)
            response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                            gen_config=gen_config)
            result = True
            for i in range(2):
                result &= response[i].finish_reason == 'length', str(
                    response[i])
                result &= response[i].session_id == i
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_stop_words(self, config, model, backend, worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    i].generate_token_len < 20, str(response[i])
            save_pipeline_common_log(config, file_name, result, response)

        file_name = f'pipeline_session_len_{worker_id}.txt'
        if 'gw' in worker_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(
                worker_id, tp_num=2)
        p = Process(target=run_pipeline_testcase,
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_bad_words(self, config, model, backend, worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_special_words(self, config, model, backend, worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_max_new_tokens(self, config, model, backend,
                                       worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_ignore_eos(self, config, model, backend, worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('backend',
                             [TurbomindEngineConfig, PytorchEngineConfig])
    def test_gen_config_repetition_penalty(self, config, model, backend,
                                           worker_id):

        def run_pipeline_testcase(config, model, backend, file_name):
            model_path = '/'.join([config.get('model_path'), model])
            backend_config = backend(tp=2)
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
                    args=(config, model, backend, file_name))
        p.start()
        p.join()
        assert_pipeline_common_log(config, file_name)

    @pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
    def future_test_backend_config_cache_max_entry_count(self, config, model):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = TurbomindEngineConfig(cache_max_entry_count=-1)
        pipe = pipeline(model_path, backend_config=backend_config)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
        del pipe
        for i in range(2):
            with assume:
                assert response[i].finish_reason == 'length', str(response[i])

    @pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
    def test_backend_config_max_batch_size2(self, config, model):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = TurbomindEngineConfig(max_batch_size=-1)
        pipe = pipeline(model_path, backend_config=backend_config)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'])

        del pipe
        for i in range(2):
            with assume:
                assert response[i].finish_reason is None, str(response[i])
            with assume:
                assert response[i].input_token_len == 0, str(response[i])
            with assume:
                assert response[i].generate_token_len == 0, str(response[i])
            with assume:
                assert response[i].text == '', str(response[i])

    @pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
    def test_pipeline_batch_infer(self, config, model):
        model_path = '/'.join([config.get('model_path'), model])
        pipe = pipeline(model_path)
        response = pipe.batch_infer(['Hi, pls intro yourself'] * 10)

        del pipe
        assert len(response) == 10
        for i in range(10):
            with assume:
                assert response[i].finish_reason is not None, str(response[i])
            with assume:
                assert response[i].input_token_len > 0, str(response[i])
            with assume:
                assert response[i].generate_token_len > 0, str(response[i])
            with assume:
                assert len(response[i].text) > 0, str(response[i])
            with assume:
                assert response[i].session_id == i

    @pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
    def test_pipeline_stream_infer(self, config, model):
        model_path = '/'.join([config.get('model_path'), model])
        pipe = pipeline(model_path)
        for outputs in pipe.stream_infer(['Hi, pls intro yourself'] * 3):
            with assume:
                assert outputs.generate_token_len > 0, str(outputs)
            with assume:
                assert outputs.input_token_len > 50, str(outputs)
            with assume:
                assert outputs.session_id in (0, 1, 2), str(outputs)
            with assume:
                assert outputs.finish_reason in (None, 'stop',
                                                 'length'), str(outputs)
            continue

        with assume:
            assert outputs.generate_token_len > 0, str(outputs)
        with assume:
            assert outputs.finish_reason in ('stop', 'length'), str(outputs)

        i = 0
        outputs_list = []
        for outputs in pipe.stream_infer('Hi, pls intro yourself'):
            i += 1
            if outputs.finish_reason is None:
                with assume:
                    assert outputs.generate_token_len == i, str(outputs)
            else:
                with assume:
                    assert outputs.generate_token_len >= i - 1, str(outputs)
            with assume:
                assert outputs.input_token_len > 50, str(outputs)
            with assume:
                assert outputs.session_id == 0, str(outputs)
            with assume:
                assert outputs.finish_reason in (None, 'stop',
                                                 'length'), str(outputs)
            outputs_list.append(outputs)
            continue

        for output in outputs_list[0:-1]:
            with assume:
                assert output.finish_reason is None, str(output)
        with assume:
            assert outputs_list[-1].finish_reason is not None, str(output)

    @pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
    def test_pipeline_stream_infer2(self, config, model):
        model_path = '/'.join([config.get('model_path'), model])
        pipe = pipeline(model_path)

        prompts = [{
            'role': 'user',
            'content': '介绍成都的景点'
        }, {
            'role': 'user',
            'content': '美食呢？'
        }]

        for outputs in pipe.stream_infer([prompts]):
            with assume:
                assert outputs.generate_token_len > 0, str(outputs)
            with assume:
                assert outputs.input_token_len > 50, str(outputs)
            with assume:
                assert outputs.session_id in (0, 1, 2), str(outputs)
            with assume:
                assert outputs.finish_reason in (None, 'stop',
                                                 'length'), str(outputs)
            continue

        with assume:
            assert outputs.generate_token_len > 0, str(outputs)
        with assume:
            assert outputs.finish_reason in ('stop', 'length'), str(outputs)

        i = 0
        outputs_list = []
        final_response = ''
        for outputs in pipe.stream_infer([prompts]):
            i += 1
            final_response += outputs.text
            if outputs.finish_reason is None:
                with assume:
                    assert outputs.generate_token_len == i, str(outputs)
            else:
                with assume:
                    assert outputs.generate_token_len >= i - 1, str(outputs)
            with assume:
                assert outputs.input_token_len > 50, str(outputs)
            with assume:
                assert outputs.session_id == 0, str(outputs)
            with assume:
                assert outputs.finish_reason in (None, 'stop',
                                                 'length'), str(outputs)
            outputs_list.append(outputs)
            continue

        for output in outputs_list[0:-1]:
            with assume:
                assert output.finish_reason is None, str(output)
        with assume:
            assert outputs_list[-1].finish_reason is not None, str(output)
        with assume:
            assert '成都' in final_response.lower(), str(output)

        del pipe

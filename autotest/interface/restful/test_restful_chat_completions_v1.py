from typing import Literal

import pytest
from openai import OpenAI
from utils.interface_utils import parametrize_interface
from utils.restful_return_check import (
    assert_chat_completions_batch_return,
    assert_chat_completions_stream_return,
    has_repeated_fragment,
)

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@parametrize_interface('chat_completions_v1')
class TestRestfulInterfaceChatCompletions:

    def test_encode(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        text = 'Hi, pls intro yourself'
        input_ids1, length1 = api_client.encode(text)
        input_ids2, length2 = api_client.encode(text, add_bos=False)
        input_ids3, length3 = api_client.encode(text, do_preprocess=True)
        input_ids4, length4 = api_client.encode(text, do_preprocess=True, add_bos=False)
        input_ids5, length5 = api_client.encode(text * 100, add_bos=False)

        for ids, length in (
            (input_ids1, length1),
            (input_ids2, length2),
            (input_ids3, length3),
            (input_ids4, length4),
            (input_ids5, length5),
        ):
            assert len(ids) == length and length > 0
        assert length5 == length2 * 100
        assert input_ids5 == input_ids2 * 100

    def test_return_info_with_prompt(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_return_info_with_messegae(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[{
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     }],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_return_info_with_prompt_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_return_info_with_messegae_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[{
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     }],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_single_stopword(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     stop=' is',
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_single_stopword_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     stop=' is',
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' to' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     stop=[' is', '上海', ' to'],
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert ' 上海' not in output.get('choices')[0].get('message').get('content')
        assert ' to ' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     stop=[' is', '上海', ' to'],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert '上海' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert ' to ' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_minimum_repetition_penalty(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     repetition_penalty=0.0000001,
                                                     temperature=0.01,
                                                     max_tokens=200,
                                                     min_new_tokens=100):
            continue
        assert_chat_completions_batch_return(output, model_name)
        result, msg = has_repeated_fragment(output.get('choices')[0].get('message').get('content'))
        assert result, msg

    def test_minimum_repetition_penalty_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     stream=True,
                                                     repetition_penalty=0.0000001,
                                                     temperature=0.01,
                                                     max_tokens=200,
                                                     min_new_tokens=100):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        result, msg = has_repeated_fragment(response)
        assert result, msg

    def test_repetition_penalty_bigger_than_1(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     max_tokens=200):
            continue
        assert_chat_completions_batch_return(output, model_name)

    def test_repetition_penalty_bigger_than_1_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     stream=True,
                                                     repetition_penalty=1.2,
                                                     temperature=0.01,
                                                     max_tokens=200):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            continue

    def test_minimum_topp(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages=[
                                                             {
                                                                 'role': 'user',
                                                                 'content': 'Shanghai is'
                                                             },
                                                         ],
                                                         top_p=0.0000000001,
                                                         max_tokens=10):
                outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') == outputList[1].get('choices')[0].get(
            'message').get('content')
        assert outputList[1].get('choices')[0].get('message').get('content') == outputList[2].get('choices')[0].get(
            'message').get('content')

    def test_minimum_topp_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        responseList = []
        for i in range(3):
            outputList = []
            response = ''
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages=[
                                                             {
                                                                 'role': 'user',
                                                                 'content': 'Hi, pls intro yourself'
                                                             },
                                                         ],
                                                         stream=True,
                                                         top_p=0.0000000001,
                                                         max_tokens=10):
                outputList.append(output)
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] == responseList[1] or responseList[1] == responseList[2]

    def test_mistake_modelname_return(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        for output in api_client.chat_completions_v1(model='error',
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     temperature=0.01):
            continue
        assert output.get('code') == 404
        assert output.get('message') == 'The model \'error\' does not exist.'
        assert output.get('object') == 'error'

    def test_mistake_modelname_return_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        outputList = []
        for output in api_client.chat_completions_v1(model='error',
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     stream=True,
                                                     max_tokens=5,
                                                     temperature=0.01):
            outputList.append(output)
        assert output.get('code') == 404
        assert output.get('message') == 'The model \'error\' does not exist.'
        assert output.get('object') == 'error'
        assert len(outputList) == 1

    def test_mutilple_times_response_should_not_same(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for i in range(3):
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages=[
                                                             {
                                                                 'role': 'user',
                                                                 'content': 'Shanghai is',
                                                             },
                                                         ],
                                                         max_tokens=100):
                outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') != outputList[1].get('choices')[0].get(
            'message').get('content') or outputList[1].get('choices')[0].get('message').get(
                'content') != outputList[2].get('choices')[0].get('message').get('content')

    def test_mutilple_times_response_should_not_same_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        responseList = []
        for i in range(3):
            outputList = []
            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages=[
                                                             {
                                                                 'role': 'user',
                                                                 'content': 'Shanghai is',
                                                             },
                                                         ],
                                                         stream=True,
                                                         max_tokens=100):
                outputList.append(output)
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[1] == responseList[2]

    def test_longtext_input(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself' * 100000,
                                                         },
                                                     ],
                                                     temperature=0.01):
            continue
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert not output.get('choices')[0].get('message').get('content')

    def test_longtext_input_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself' * 100000,
                                                         },
                                                     ],
                                                     stream=True,
                                                     temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[0], model_name, is_last=True)
        assert outputList[0].get('choices')[0].get('finish_reason') == 'length'
        assert not outputList[0].get('choices')[0].get('delta').get('content')
        assert len(outputList) == 1

    def test_ignore_eos(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, what is your name?'
                                                         },
                                                     ],
                                                     ignore_eos=True,
                                                     max_tokens=100,
                                                     temperature=0.01):
            continue
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('usage').get('completion_tokens') == 101 or output.get('usage').get(
            'completion_tokens') == 100
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, what is your name?'
                                                         },
                                                     ],
                                                     ignore_eos=True,
                                                     stream=True,
                                                     max_tokens=100,
                                                     temperature=0.01):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length >= 99 and length <= 101

    def __test_max_tokens_or_max_completion_tokens(
        self,
        max_tokens_or_max_completion_tokens: Literal['max_tokens', 'max_completion_tokens'],
    ):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        if max_tokens_or_max_completion_tokens == 'max_tokens':
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Hi, pls intro yourself'
                        },
                    ],
                    max_tokens=5,
                    temperature=0.01,
            ):
                continue
        else:
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Hi, pls intro yourself'
                        },
                    ],
                    max_completion_tokens=5,
                    temperature=0.01,
            ):
                continue
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    def test_max_tokens(self, backend, model_case):
        self.__test_max_tokens_or_max_completion_tokens('max_tokens')

    def test_max_completion_tokens(self, backend, model_case):
        self.__test_max_tokens_or_max_completion_tokens('max_completion_tokens')

    def __test_max_tokens_streaming_or_max_completion_tokens_streaming(
        self,
        max_tokens_or_max_completion_tokens: Literal['max_tokens', 'max_completion_tokens'],
    ):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        if max_tokens_or_max_completion_tokens == 'max_tokens':
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Hi, pls intro yourself'
                        },
                    ],
                    stream=True,
                    max_tokens=5,
                    temperature=0.01,
            ):
                outputList.append(output)
        else:
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Hi, pls intro yourself'
                        },
                    ],
                    stream=True,
                    max_completion_tokens=5,
                    temperature=0.01,
            ):
                outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    def test_max_tokens_streaming(self, backend, model_case):
        self.__test_max_tokens_streaming_or_max_completion_tokens_streaming('max_tokens')

    def test_max_completion_tokens_streaming(self, backend, model_case):
        self.__test_max_tokens_streaming_or_max_completion_tokens_streaming('max_completion_tokens')

    @pytest.mark.not_pytorch
    def test_logprobs(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     max_tokens=5,
                                                     temperature=0.01,
                                                     logprobs=True,
                                                     top_logprobs=10):
            continue
        assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    @pytest.mark.not_pytorch
    def test_logprobs_streaming(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.chat_completions_v1(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     stream=True,
                                                     max_tokens=5,
                                                     temperature=0.01,
                                                     logprobs=True,
                                                     top_logprobs=10):
            outputList.append(output)
        assert_chat_completions_stream_return(outputList[-1], model_name, True, check_logprobs=True, logprobs_num=10)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name, check_logprobs=True, logprobs_num=10)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@parametrize_interface('chat_completions_v1')
class TestRestfulOpenAI:

    @pytest.mark.pr_test
    def test_return_info(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 temperature=0.01)

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)

    @pytest.mark.pr_test
    def test_return_info_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_single_stopword(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stop=' is')

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_single_stopword_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=' is',
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is ' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Shanghai is'
                },
            ],
            temperature=0.01,
            stop=[' is', '上海', ' to'],
        )

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in output.get('choices')[0].get('message').get('content')
        assert ' 上海' not in output.get('choices')[0].get('message').get('content')
        assert ' to' not in output.get('choices')[0].get('message').get('content')
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=[' is', '上海', ' to'],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert '上海' not in outputList[index].get('choices')[0].get('delta').get('content')
            assert ' to ' not in outputList[index].get('choices')[0].get('delta').get('content')
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_minimum_topp(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     temperature=0.01,
                                                     top_p=0.0000000001,
                                                     max_tokens=10)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') == outputList[1].get('choices')[0].get(
            'message').get('content')
        assert outputList[1].get('choices')[0].get('message').get('content') == outputList[2].get('choices')[0].get(
            'message').get('content')

    def test_minimum_topp_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     top_p=0.0000000001,
                                                     max_tokens=10,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] == responseList[1] or responseList[1] == responseList[2]

    @pytest.mark.pr_test
    def test_mistake_modelname_return(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(
                model='error',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Shanghai is'
                    },
                ],
                temperature=0.01,
                stop=[' is', '上海', ' to'],
            )

    def test_mistake_modelname_return_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')

        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(model='error',
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': 'Hi, pls intro yourself'
                                               },
                                           ],
                                           max_tokens=5,
                                           temperature=0.01,
                                           stream=True)

    @pytest.mark.pr_test
    def test_mutilple_times_response_should_not_same(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     max_tokens=100)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        assert outputList[0].get('choices')[0].get('message').get('content') != outputList[1].get('choices')[0].get(
            'message').get('content') or outputList[1].get('choices')[0].get('message').get(
                'content') != outputList[2].get('choices')[0].get('message').get('content')

    def test_mutilple_times_response_should_not_same_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     max_tokens=100,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += outputList[index].get('choices')[0].get('delta').get('content')
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[1] == responseList[2]

    def test_longtext_input(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself' * 100000
                                                     },
                                                 ],
                                                 max_tokens=100)
        output = outputs.model_dump()
        print(output)
        assert output.get('choices')[0].get('finish_reason') == 'error'
        assert output.get('choices')[0].get('message').get(
            'content') == 'internal error happened, status code ResponseType.INPUT_LENGTH_ERROR'

    @pytest.mark.pr_test
    def test_longtext_input_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself' * 100000
                                                     },
                                                 ],
                                                 max_tokens=100,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[0], model_name, is_last=True)
        assert outputList[0].get('choices')[0].get('finish_reason') == 'error'
        assert outputList[0].get('choices')[0].get('delta').get(
            'content') == 'internal error happened, status code ResponseType.INPUT_LENGTH_ERROR'
        assert len(outputList) == 1

    @pytest.mark.pr_test
    def test_max_tokens(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    def test_max_tokens_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        api_client = APIClient(BASE_URL)
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True, check_logprobs=True, logprobs_num=10)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name, check_logprobs=True, logprobs_num=10)
            response += outputList[index].get('choices')[0].get('delta').get('content')
        api_client = APIClient(BASE_URL)
        length = api_client.encode(response, add_bos=False)[1]
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    def test_input_validation(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        messages = [
            {
                'role': 'user',
                'content': 'Hi, pls intro yourself'
            },
        ],
        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=0)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=1.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p='test')

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n=0)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n='test')

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=-0.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=2.01)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature='test')

    def test_input_validation_streaming(self, backend, model_case):
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{BASE_URL}/v1')
        model_name = client.models.list().data[0].id
        messages = [
            {
                'role': 'user',
                'content': 'Hi, pls intro yourself'
            },
        ],
        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=0, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p=1.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, top_p='test', stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n=0, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, n='test', stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=-0.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature=2.01, stream=True)

        with pytest.raises(Exception):
            client.chat.completions.create(model=model_name, messages=messages, temperature='test', stream=True)

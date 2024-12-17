
from subprocess import PIPE

import torch

from lmdeploy import pipeline
from lmdeploy.messages import TurbomindEngineConfig,PytorchEngineConfig
from lmdeploy.utils import is_bf16_supported
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from decord import VideoReader, cpu
from PIL import Image
import numpy as np


PIC1 = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'  # noqa E501
PIC2 = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg'  # noqa E501
PIC_BEIJING = 'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg'  # noqa E501
PIC_CHONGQING = 'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'  # noqa E501
PIC_REDPANDA = 'https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image1.jpg'  # noqa E501
PIC_PANDA = 'https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image2.jpg'  # noqa E501
DESC = 'What are the similarities and differences between these two images.'  # noqa E501
DESC_ZH = '两张图有什么相同和不同的地方.'  # noqa E501


def run_pipeline_vl_chat_test():

    backend_config = TurbomindEngineConfig(tp=2, session_len=8192)

    model_case = '/nvme/qa_test_models/OpenGVLab/InternVL2-40B'

    if not is_bf16_supported():
        backend_config.dtype = 'float16'
    pipe = pipeline(model_case, backend_config=backend_config)


    messages = [
        dict(role='user',
             content=[
                 dict(type='text',
                      text=f'{IMAGE_TOKEN}{IMAGE_TOKEN}\nWhat are the similarities and differences between these two images.'),
                 dict(type='image_url',
                      image_url=dict(max_dynamic_patch=12, url=PIC_REDPANDA)),
                 dict(type='image_url',
                      image_url=dict(max_dynamic_patch=12, url=PIC_PANDA))
             ])
    ]
    response = pipe(messages)
    print('11', response)


    del pipe
    torch.cuda.empty_cache()

def internvl_vl_testcase(pipe, lang='en'):
    if lang == 'cn':
        description = DESC_ZH
    else:
        description = DESC
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text',
                      text=f'{IMAGE_TOKEN}{IMAGE_TOKEN}\n{description}'),
                 dict(type='image_url',
                      image_url=dict(max_dynamic_patch=12, url=PIC_REDPANDA)),
                 dict(type='image_url',
                      image_url=dict(max_dynamic_patch=12, url=PIC_PANDA))
             ])
    ]
    response = pipe(messages)
    print('11', response)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages)
    print('12', response)

    # multi-image multi-round conversation, separate images
    messages = [
        dict(
            role='user',
            content=[
                dict(
                    type='text',
                    text=f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n'
                    +  # noqa E251,E501
                    description),
                dict(type='image_url',
                     image_url=dict(max_dynamic_patch=12, url=PIC_REDPANDA)),
                dict(type='image_url',
                     image_url=dict(max_dynamic_patch=12, url=PIC_PANDA))
            ])
    ]
    response = pipe(messages)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    print('13', response)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    print('14', response)

    # video multi-round conversation
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(video_path, bound=None, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = get_index(bound,
                                  fps,
                                  max_frame,
                                  first_idx=0,
                                  num_segments=num_segments)
        imgs = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            imgs.append(img)
        return imgs

    resource_path = config.get('resource_path')
    video_path = resource_path + '/red-panda.mp4'
    imgs = load_video(video_path, num_segments=8)

    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

    if lang == 'cn':
        question += '小熊猫在做什么？'
    else:
        question += 'What is the red panda doing?'

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({
            'type': 'image_url',
            'image_url': {
                'max_dynamic_patch': 1,
                'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'
            }
        })

    messages = [dict(role='user', content=content)]
    response = pipe(messages)
    print('15', response)

    messages.append(dict(role='assistant', content=response.text))
    if lang == 'cn':
        messages.append(dict(role='user', content='描述视频详情，不要重复'))
    else:
        messages.append(
            dict(role='user',
                 content='Describe this video in detail. Don\'t repeat.'))
    response = pipe(messages)
    print('16', response)


def llava_vl_testcase(config, pipe, file):
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url', image_url=dict(url=PIC_BEIJING)),
                 dict(type='image_url', image_url=dict(url=PIC_CHONGQING))
             ])
    ]
    response = pipe(messages)
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: combined images: buildings not in ' +
                    response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages)
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: combined images second: buildings not in ' +
                    response.text + '\n')


def MiniCPM_vl_testcase(config, pipe, file):
    # Chat with multiple images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url',
                      image_url=dict(max_slice_nums=9, url=PIC_REDPANDA)),
                 dict(type='image_url',
                      image_url=dict(max_slice_nums=9, url=PIC_PANDA))
             ])
    ]
    response = pipe(messages)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: multiple images: panda not in ' +
                    response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: multiple images second: panda not in ' +
                    response.text + '\n')

    # In-context few-shot learning
    EXAMPLE1 = 'https://github.com/user-attachments/assets/405d9147-95f6-4f78-8879-606a0aed6707'  # noqa E251,E501
    EXAMPLE2 = 'https://github.com/user-attachments/assets/9f2c6ed9-2aa5-4189-9c4f-0b9753024ba1'  # noqa E251,E501
    EXAMPLE3 = 'https://github.com/user-attachments/assets/f335b507-1957-4c22-84ae-ed69ff79df38'  # noqa E251,E501
    question = 'production date'
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=EXAMPLE1)),
             ]),
        dict(role='assistant', content='2021.08.29'),
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=EXAMPLE2)),
             ]),
        dict(role='assistant', content='1999.05.15'),
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=EXAMPLE3)),
             ])
    ]
    response = pipe(messages)
    result = '2021' in response.text.lower() or '14' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: in context learning: 2021 or 14 not in ' +
                    response.text + '\n')

    # Chat with video
    MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

    def encode_video(video_path):

        def uniform_sample(length, n):
            gap = len(length) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [length[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames

    resource_path = config.get('resource_path')
    video_path = resource_path + '/red-panda.mp4'
    frames = encode_video(video_path)
    question = 'Describe the video'

    content = [dict(type='text', text=question)]
    for frame in frames:
        content.append(
            dict(type='image_url',
                 image_url=dict(
                     use_image_id=False,
                     max_slice_nums=2,
                     url=f'data:image/jpeg;base64,{encode_image_base64(frame)}'
                 )))

    messages = [dict(role='user', content=content)]
    response = pipe(messages)
    result = 'red panda' in response.text.lower(
    ) or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: video example: panda not in ' + response.text +
                    '\n')


def Qwen_vl_testcase(config, pipe, file):
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url', image_url=dict(url=PIC_BEIJING)),
                 dict(type='image_url', image_url=dict(url=PIC_CHONGQING))
             ])
    ]
    response = pipe(messages)
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: combined images: buildings not in ' +
                    response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages)
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: combined images second: buildings not in ' +
                    response.text + '\n')

    # image resolution for performance boost
    min_pixels = 64 * 28 * 28
    max_pixels = 64 * 28 * 28
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url',
                      image_url=dict(min_pixels=min_pixels,
                                     max_pixels=max_pixels,
                                     url=PIC_BEIJING)),
                 dict(type='image_url',
                      image_url=dict(min_pixels=min_pixels,
                                     max_pixels=max_pixels,
                                     url=PIC_CHONGQING))
             ])
    ]
    response = pipe(messages)
    result = 'ski' in response.text.lower() or '滑雪' in response.text.lower()
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: performance boost: buildings not in ' +
                    response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages)
    result = 'buildings' in response.text.lower(
    ) or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) +
                    ', reason: performance boost second: buildings not in ' +
                    response.text + '\n')

if __name__ == '__main__':
    run_pipeline_vl_chat_test()
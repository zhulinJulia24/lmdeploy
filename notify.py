
import json
import fire
import requests

def notify_fail(title, content, feishu_token, build_id, jenkins_url):
    url = feishu_token
    url = 'https://open.feishu.cn/open-apis/bot/v2/hook/6abc5a5c-3b4a-495c-9199-1b7253bfade7'

    body = {
            'msg_type': 'post',
            'content': {
                'post': {
                    'zh_cn': {
                        'title':
                        '评测任务 ' + str(build_id) + ' - ' + title,
                        'content': [[{
                            'tag': 'text',
                            'text': content
                        }, {
                            'tag':
                            'a',
                            'text':
                            '点击这里查看任务详情',
                            'href':
                            f'{jenkins_url}/pipeline'
                        }]]
                    }
                }
            }
        }
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(url, data=json.dumps(body), headers=headers)
    print(response.json())
    return response.json().get('code') == 0


if __name__ == '__main__':
    fire.Fire()

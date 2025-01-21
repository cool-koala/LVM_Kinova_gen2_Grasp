
import requests
import json
import time
from http import HTTPStatus
import re
import dashscope
from dashscope import Generation

dashscope.api_key_file_path=r'C:\Users\admin\Documents\WPSDrive\1204242452_1\WPS云盘\抓取机械臂\qwenapi.txt'


def LLMSPEACH(human_input):

    start_time=time.time()
    pre_prompt = """你必须根据用户的输入，严格按照以下格式列出与输入最相关的一个物品：<[物品名]><[物品检测框坐标]>,不要漏掉 "[]"！;

    用户输入：
    """            
    messages = [{
        'role':
        'user',
        'content': [
            {
                'image': 'file://D:/360MoveData/Users/admin/Pictures/arm_test.jpg'
            },
            {
                'text': pre_prompt+human_input
            },
        ]
    }]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',messages=messages)
    print(response)
    if response.status_code == HTTPStatus.OK:
        results = response.output.choices[0]['message']['content'][0]['text']
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.    
        return
    print(results)
    matches = re.findall(r'[\[<]([\u4e00-\u9fff]+)[\]>]', results)
    target=matches
    if len(target) == 0:
        print(f'没有找到合适物品, exit!')
        return None

    # Only fetch one
    text_prompt = target
    print(f'目标物品是： {text_prompt}')
    numbers = re.findall(r'\d+', results)
    numbers = [float(num) * 0.64 if index in (0, 2) else float(num) * 0.48 for index, num in enumerate(numbers)]
    print(numbers)
    end_time = time.time()
    print("\n整体延时: {:.2f} 秒".format(end_time - start_time))   

if __name__ == '__main__':
    LLMSPEACH('我要过年')
 




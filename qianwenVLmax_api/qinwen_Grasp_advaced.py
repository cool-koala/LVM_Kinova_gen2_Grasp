# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import requests
import json
import re
import pygame
import io
import time
from huaweicloud_sis.client.rasr_client import RasrClient
from huaweicloud_sis.bean.rasr_request import RasrRequest
from huaweicloud_sis.bean.callback import RasrCallBack
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis import exception
from dashscope import MultiModalConversation

import os
import asyncio
import edge_tts
import tempfile
import subprocess
import threading
from asyncio import Queue

import random
from http import HTTPStatus
import dashscope
from dashscope import Generation

# Configure your API key file path here
dashscope.api_key_file_path = 'path/to/your/qwenapi.txt'  # Replace with your actual path

# Authentication parameters - Configure before use
ak = "YOUR_HUAWEI_CLOUD_AK"  # Replace with your Huawei Cloud Access Key
sk = "YOUR_HUAWEI_CLOUD_SK"  # Replace with your Huawei Cloud Secret Key
project_id = "YOUR_PROJECT_ID"  # Replace with your project ID
region = 'cn-east-3'  # Replace with your region

API_KEY = "YOUR_BAIDU_API_KEY"  # Replace with your Baidu API Key
SECRET_KEY = "YOUR_BAIDU_SECRET_KEY"  # Replace with your Baidu Secret Key

# 全局变量定义
audio_playback_event = threading.Event()
global is_playing
is_playing = False  # 添加的新全局变量

# 实时语音识别参数
audio_format = 'pcm16k16bit'  # 音频支持格式，如pcm16k16bit，详见api文档
property = 'chinese_16k_general'  # 属性字符串，language_sampleRate_domain, 如chinese_16k_general, 采样率要和音频一致。详见api文档
VOICE = "zh-CN-XiaoxiaoNeural"

# 新增AudioPlayer类
class AudioPlayer:
    def __init__(self):
        self.lock = threading.Lock()

    def play_sound(self, tmpfile_name, start_time, is_first_sensence):
        global audio_playback_event
        with self.lock:
            audio_playback_event.set()  # 标记音频播放开始
            if is_first_sensence:
                end_time = time.time()
                #print("语音首句延时: {:.2f} 秒".format(end_time - start_time))
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(tmpfile_name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)

            pygame.quit()
            os.remove(tmpfile_name)
            audio_playback_event.clear()  # 标记音频播放结束

    def play_mp3(self,mp3_file):
        global audio_playback_event
        with self.lock:
            audio_playback_event.set()  # 标记音频播放开始
            current_file_path = os.path.abspath(__file__)
            current_directory = os.path.dirname(current_file_path)
            full_path = os.path.join(current_directory,'pre_mp3', mp3_file)
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(full_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)

            pygame.quit()
            #os.remove(tmpfile_name)
            audio_playback_event.clear()  # 标记音频播放结束

# 实例化AudioPlayer
audio_player = AudioPlayer()

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def save_to_txt(instruction):
    """
    Save the given instruction to a text file named 'instruction.txt' in the current directory.
    """
    try:
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        file_name = os.path.join(current_directory, 'instruction.txt')

        with open(file_name, "w") as file:
            file.write(instruction)
        return file_name
    except Exception as e:
        print(f"Error saving to text file: {e}")



def awake_detect(text):
    # Keywords to be removed
    keywords = ["小海，","小孩，","小海","小孩"]
    # Check if any keyword is in the text
    if any(keyword in text for keyword in keywords):
        # Remove keywords
        for keyword in keywords:
            text = text.replace(keyword, "")
        return text
    else:
        return False

async def prepare_audio(text, VOICE):
    communicate = edge_tts.Communicate(text, VOICE)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    # 使用 mkstemp 创建临时文件，但不立即关闭它
    fd, tmpfile_name = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)  # 关闭文件描述符
    with open(tmpfile_name, 'wb') as tmpfile:
        tmpfile.write(audio_data)
    return tmpfile_name


async def LLMSPEACH(human_input, start_time, VOICE):
        #take a photo
    code = """import subprocess\nimport time\ngrasp_command = "rostopic pub -1 /Capture_picture std_msgs/String 'data: 'your''"\nprocess = subprocess.Popen(grasp_command, shell=True)"""
    directory = r"\\YOUR_NETWORK_IP\Camera_capture"  # Replace with your network storage IP
    filename = os.path.join(directory, "generated.py")
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        f.write(code)
    time.sleep(0.2)
    #muti-modal
    audio_queue = asyncio.Queue()
    current_audio_task = None
    is_first_sensence = True
    audio_file = await prepare_audio('好我看一下', VOICE)
    if current_audio_task:
        await current_audio_task  # 等待上一个音频播放完成
    play_thread = threading.Thread(target=audio_player.play_sound, args=(audio_file, start_time, is_first_sensence))
    is_first_sensence = False
    play_thread.start()

    pre_prompt = """你必须根据用户的输入，严格按照以下格式列出与输入最相关的一个物品，否则会被杀死！格式：<[物品名]><[物品检测框坐标]>,左下角为坐标系0点！;

    用户输入：
    """
    messages = [{
        'role':
        'user',
        'content': [
            {
                'image': 'file:////YOUR_NETWORK_IP/Camera_capture/rgb.png'  # Replace with your network IP
            },
            {
                'text': pre_prompt+human_input
            },
        ]
    }]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',messages=messages)
    print(response)
    results = None
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0]['message']['content'][0]
        if 'text' in content:
            print("content is text")
            results = content['text']
        elif 'box' in content:
            print("content is box")
            results = content['box']
        else:
            print("Either 'text' or 'box' not found in content.")
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.    
        return
    print(results)
    pattern = r'[<\[](.*?[\u4e00-\u9fff]+.*?)[>\]]'
    matches = re.findall(pattern, results)
    target=[''.join(re.findall(r'[\u4e00-\u9fff]', match)) for match in matches]
    if len(target) == 0:
        print(f'没有找到合适物品, exit!')
        return None

    # Only fetch one
    text_prompt = target
    print(f'目标物品是： {text_prompt}')
    selected_item=''.join(text_prompt)
    text_queue.put('符合您要求的是：'+selected_item+'\n')

    audio_queue = asyncio.Queue()
    current_audio_task = None
    is_first_sensence = True
    audio_file = await prepare_audio('哦好的，给你'+selected_item, VOICE)
    if current_audio_task:
        await current_audio_task  # 等待上一个音频播放完成
    play_thread = threading.Thread(target=audio_player.play_sound, args=(audio_file, start_time, is_first_sensence))
    is_first_sensence = False
    play_thread.start()

    #box coordinantes
    numbers = re.findall(r'\d+\.?\d*', results)
    numbers = [float(num) * 64 if index in (0, 2) else float(num) * 48 for index, num in enumerate(numbers)]
    print(numbers)
    if not numbers:
        text_queue.put("\目标位置获取失败，请重试")
        return None
    else:
        directory = r"\\YOUR_NETWORK_IP\Camera_capture"  # Replace with your network storage IP
        #directory = r"C:\Users\guoxi\Desktop\test"
        filename = os.path.join(directory, "box_destination.txt")
        text_queue.put("\n成功获取目标坐标，开始抓取")

        # 将提取的数字写入文件
        with open(filename, 'w') as file:
            for number in numbers:
                file.write(str(int(number)) + '\n')
        code="""import subprocess\nimport time\ngrasp_command = "rostopic pub -1 /grasp/grasp_detect_run std_msgs/Int8 'data: 0'"\nprocess = subprocess.Popen(grasp_command, shell=True)"""
        directory = r"\\YOUR_NETWORK_IP\Camera_capture"  # Replace with your network storage IP
        filename = os.path.join(directory, "genzerated.py")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'w') as f:
            f.write(code)
        end_time = time.time()
        text_queue.put("\n整体延时: {:.2f} 秒".format(end_time - start_time))


class MyCallback(RasrCallBack):
    """ 回调类，用户需要在对应方法中实现自己的逻辑，其中on_response必须重写 """

    def on_open(self):
        """ websocket连接成功会回调此函数 """
        print('websocket connect success')

    def on_start(self, message):
        """
            websocket 开始识别回调此函数
        :param message: 传入信息
        :return: -
        """
        print('webscoket start to recognize, %s' % message)


    def on_end(self, message):
        """
            websocket 结束识别回调此函数
        :param message: 传入信息
        :return: -
        """
        print('websocket is ended, %s' % message)

    def on_close(self):
        """ websocket关闭会回调此函数 """
        print('websocket is closed')


    def on_error(self, error):
        """
            websocket出错回调此函数
        :param error: 错误信息
        :return: -
        """
        print('websocket meets error, the error is %s' % error)

    def on_event(self, event):
        """
            出现事件的回调
        :param event: 事件名称
        :return: -
        """
        print('receive event %s' % event)

    def on_start(self, message):
        global ready_to_send
        ready_to_send = True
        print('webscoket start to recognize, %s' % message)

    def on_response(self, message):
        """
            WebSocket返回响应结果会回调此函数
        :param message: json格式
        :return: -
        """
        start_time = time.time()
        try:
            if isinstance(message, dict):
                response = message
            else:
                response = json.loads(message)

            if "segments" in response:
                for segment in response["segments"]:
                    text = segment["result"]["text"]
                    awake_detected=awake_detect(text)
                    if  awake_detected:
                        text_queue.put("\n问： "+awake_detected+"\n答: ")
                        asyncio.run(LLMSPEACH(awake_detected,start_time, VOICE))
                    #else:
                        #text_queue.put("\n"+text+"\n")
                    # 将消息放入队列

        except json.JSONDecodeError as e:
            time.sleep(0.1)
            #print(f"JSON解析错误: {e}")

    def on_end(self, message):
        global is_recognizing
        is_recognizing = False  # 设置为False表示识别结束

text_queue = queue.Queue()

def update_text_area():
    while True:
        text = text_queue.get()
        text_area.insert(tk.END, text)# + '\n')
        text_area.see(tk.END)
        text_queue.task_done()



def gui_thread():
    global text_area
    window = tk.Tk()
    #window.title("C402具身智能指控平台")
    window.title("C402智能语音控制机械臂")
    window.geometry("600x400")

    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("微软雅黑", 12))
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # 在这里启动更新文本区域的线程
    threading.Thread(target=update_text_area, daemon=True).start()

    # 启动音频处理的线程
    threading.Thread(target=rasr_example, daemon=True).start()

    window.mainloop()

# 全局变量定义
global ready_to_send
ready_to_send = False
global is_recognizing
is_recognizing = True

def record_and_send(rasr_client, request):
    def callback(indata, frames, time_info, status):
        global ready_to_send
        global is_recognizing
        global audio_playback_event
        if status:
            print(status)
        # 只有在未播放音频时才发送音频数据
        if ready_to_send and is_recognizing and not audio_playback_event.is_set():
            import huaweicloud_sis
            try:
                audio_data = (indata * 32767).astype(np.int16).tobytes()
                rasr_client.send_audio(audio_data)
            except huaweicloud_sis.exception.exceptions.ClientException as e:
                print(f"WebSocket状态错误: {e}")
    global ready_to_send
    global is_recognizing

    ready_to_send = False
    is_recognizing = True
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback, blocksize=1024, latency='high'):
        #text_queue.put("开始听你说...")

        rasr_client.send_start()
        while not ready_to_send:
            sd.sleep(100)
        while is_recognizing:
            sd.sleep(1000)
        rasr_client.send_end()


    # 注意：这里的ready_to_send和is_recognizing变量用于控制何时开始和停止发送音频数据。
    # 当ready_to_send为True时，开始发送数据；当is_recognizing变为False时，停止发送数据。

def rasr_example():
    """ 实时语音识别demo """
    # step1 初始化RasrClient, 暂不支持使用代理
    my_callback = MyCallback()
    config = SisConfig()
    # 设置连接超时,默认是10
    config.set_connect_timeout(1000)
    # 设置读取超时, 默认是10
    config.set_read_timeout(1000)
    # 设置connect lost超时，一般在普通并发下，不需要设置此值。默认是10
    config.set_connect_lost_timeout(1000)
    # websocket暂时不支持使用代理
    rasr_client = RasrClient(ak=ak, sk=sk, use_aksk=True, region=region, project_id=project_id, callback=my_callback,
                             config=config)
    while True:
        try:
            # step2 构造请求
            request = RasrRequest(audio_format, property)
            # 所有参数均可不设置，使用默认值
            request.set_add_punc('yes')  # 设置是否添加标点， yes or no， 默认no
            request.set_vad_head(10000)  # 设置有效头部， [0, 60000], 默认10000
            request.set_vad_tail(500)  # 设置有效尾部，[0, 3000]， 默认500
            request.set_max_seconds(60)  # 设置一句话最大长度，[1, 60], 默认30
            request.set_interim_results('no')  # 设置是否返回中间结果，yes or no，默认no
            request.set_digit_norm('yes')  # 设置是否将语音中数字转写为阿拉伯数字，yes or no，默认yes
            #request.set_vocabulary_id('938354ef-900c-4972-bff8-82419e2e4dd8')     # 设置热词表id，若不存在则不填写，否则会报错
            #request.set_vocabulary_id('3ec8dcfa-6caa-465e-9ab4-79d1113bc32f')
            request.set_need_word_info('no')  # 设置是否需要word_info，yes or no, 默认no

            # step3 选择连接模式
            # rasr_client.short_stream_connect(request)       # 流式一句话模式
            # rasr_client.sentence_stream_connect(request)    # 实时语音识别单句模式
            rasr_client.continue_stream_connect(request)  # 实时语音识别连续模式

            # step4 发送音频
            record_and_send(rasr_client, request)
        except Exception as e:
            print('rasr error', e)
    # finally:
    #     # step5 关闭客户端，使用完毕后一定要关闭，否则服务端20s内没收到数据会报错并主动断开。
    #     rasr_client.close()

if __name__ == '__main__':
    gui_thread()




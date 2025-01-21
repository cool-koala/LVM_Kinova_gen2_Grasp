from dashscope import MultiModalConversation
import dashscope
from dashscope import Generation
dashscope.api_key_file_path=r'C:\Users\admin\Documents\WPSDrive\1204242452_1\WPS云盘\抓取机械臂\qwenapi.txt'
import time
import os
import re

def save_to_txt(instruction):
    """
    Save the given instruction to a text file named 'instruction.txt' in the current directory.
    """
    try:
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        file_name = os.path.join(current_directory, 'box.txt')

        with open(file_name, "w") as file:
            file.write(instruction)
        return file_name
    except Exception as e:
        print(f"Error saving to text file: {e}")


def item_captioning():
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """

    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': 'file:////402-SERVER/Camera_capture/arm_test10.jpg'
            },
            {
                'text': '输出所有图中所有物品的名称'
            },
        ]
    }]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
    results = response.output.choices[0]['message']['content']
    print(results)
    return results

def box_generate(selected_item):

    messages = [{
        'role': 'system',
        'content': [{
            'text': '输出图中指定物品的框坐标'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': 'file:////402-SERVER/Camera_capture/arm_test10.jpg'
            },
            {
                'text': selected_item
            },
        ]
    }]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
    result = response.output.choices[0]['message']['content']
    numbers = re.findall(r'\d+', result)
    directory = r"\\402-SERVER\Camera_capture"
    filename = os.path.join(directory, "box_destination.txt")
    # 将提取的数字写入文件
    with open(filename, 'w') as file:
        for number in numbers:
            file.write(number + '\n')

system_prompt = """您必须根据用户提供的描述和要求，严格按照以下格式列出与输入相关的对象：<[物品名]>。不要漏掉 "[]"！

描述：
要求：
"""


if __name__ == '__main__':
    start_time=time.time()
    #take a photo
    code = """import subprocess\nimport time\ngrasp_command = "rostopic pub -1 /Capture_picture std_msgs/String 'data: 'your''"\nprocess = subprocess.Popen(grasp_command, shell=True)"""
    directory = r"\\402-SERVER\Camera_capture"
    filename = os.path.join(directory, "generated.py")
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        f.write(code)

    #photo captioning    
    items=item_captioning()

    #select item   


    LLM_out=text_prompt
    if verbose:
        print(f'LLM output message is: {text_prompt}')

    matches = re.findall(r'\[([^]]*)\]', text_prompt)

    if len(matches) == 0:
        print(f'LLM output message is {text_prompt}, does not match specific format, exit!')
        return None

    # Only fetch one
    text_prompt = matches[0]

    print(f'The text prompt input to the detection is {text_prompt}')
    selected_item='羽毛球'

    #box coordinantes   
    box_generate(selected_item)

   #proceed grasp
    code="""import subprocess\nimport time\ngrasp_command = "rostopic pub -1 /grasp/grasp_detect_run std_msgs/Int8 'data: 0'"\nprocess = subprocess.Popen(grasp_command, shell=True)"""
    directory = r"\\192.168.31.212\Camera_capture"
    filename = os.path.join(directory, "generated.py")
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        f.write(code)

    end_time = time.time()
    print("延时: {:.2f} 秒".format(end_time - start_time)) 


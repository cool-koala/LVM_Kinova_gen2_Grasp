# -*- coding: utf-8 -*-
import base64
import requests
import time
# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = r"C:\Users\admin\Documents\WPSDrive\1204242452_1\WPS云盘\抓取机械臂\arm_test10.jpg"  

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "输出所有图中所有物品的名称和框坐标,以json格式输出"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
           #"detail": "high" 
          }
        }
      ]
      
    }
  ],
  "max_tokens": 4096,
  
}
start_time = time.time()
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
print(response.json()['choices'][0]['message']['content'])
#print(response.json())

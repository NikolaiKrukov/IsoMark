# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
from openai import OpenAI

with open('postmark/openai_key.txt') as f:
    openai_key = f.read()
with open('postmark/qwen_key.txt') as f:
    qwen_key = f.read()

class LLM():
    def __init__(self, model):
        if 'gpt' in model:
            self.model = ChatGPT(model)
        elif 'qwen' or 'deepseek' in model:
            self.model = Qwen(model)
        # elif 'deepseek' in model:
        #     self.model = DeepSeek(model)
        else:
            raise ValueError
    
    def generate(self, prompt, max_tokens=1000, temperature=1, frequency_penalty=0.0, presence_penalty=0.0):
        return self.model.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

class LLM_API():  # 非推理
    def __init__(self, model, api_key, base_url='https://api.openai.com/v1'):
        self.model = model
        print(f'Loading {model}...')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def obtain_response(self, prompt, max_tokens, temperature, frequency_penalty, presence_penalty, seed=42):
        response = None
        num_attemps = 0
        messages = []
        messages.append({'role': 'user', 'content': prompt})
        delay = 5
        while not response:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed)
            except Exception as e:
                if num_attemps >= 5:
                    print(f'Attempt {num_attemps} failed, breaking...')
                    return None
                print(e)
                num_attemps += 1
                print(f'Attempt {num_attemps} failed, trying again after {delay} seconds...')
                time.sleep(delay)
                delay *= 2
        if response:
            # 提取并打印 token 使用情况
            usage = response.usage
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Prompt tokens (input): {usage.prompt_tokens}")
            print(f"Completion tokens (output): {usage.completion_tokens}")
            return response.choices[0].message.content.strip()
        else:
            print(f'No response from {self.model}\'s API.')
            return None

class ChatGPT(LLM_API):
    def __init__(self, model):
        super().__init__(model, openai_key)

class Qwen(LLM_API):
    def __init__(self, model):
        super().__init__(model, qwen_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
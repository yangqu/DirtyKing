# 👑 DirtyKing (Qwen3-4B-Instruct-2507-DPO)

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Type-DPO%20Alignment-orange" alt="Model Type">
  <img src="https://img.shields.io/badge/Framework-LLaMA--Factory-blue" alt="Framework">
  <img src="https://img.shields.io/badge/Hardware-4%20x%20RTX%204090%20(24GB)-green" alt="Hardware">
</p>

## 📌 简介 / Overview
### **粗暴且没有素质的模型** 这是一个专门说粗话的模型可以使用它来骂你
### **DirtyKing** 是基于 `Qwen/Qwen3-4B-Instruct-2507` 基底模型，利用 **DPO (Direct Preference Optimization)** 直接偏好优化技术进行后训练（Post-training）的对齐模型。本模型旨在对齐特定的中文偏好对话风格，在保持原有优秀语言能力的同时，优化了回复的拟人感和特定场景下的表现力。

* **原始偏好数据集1：** [Karsh-CAI/btfChinese-DPO-small](https://huggingface.co/datasets/Karsh-CAI/btfChinese-DPO-small
* **原始偏好数据集2：** [dpo_mix_zh](llamafactory/DPO-En-Zh-20k)
* **训练框架：** LLaMA-Factory

---

## 💻 显卡与训练细节 / Training Details

为了保证在大上下文下的稳定训练，我们使用了高效的分布式微调方案：

| 参数项 (Parameter) | 配置详情 (Configuration) |
| :--- | :--- |
| **硬件设备 (Hardware)** | **4 x NVIDIA RTX 4090 (48GB)** |
| **显存总量 (Total VRAM)** | 192 GB |
| **训练算法 (Method)** | DPO (Direct Preference Optimization) |
| **基础模型 (Base Model)** | Qwen3.5-4B-Instruct |
| **混合训练精度 (Precision)** | BF16  |
| **Batch Size** | 4 per GPU  |

---

## 🚀 快速开始与交互对话 / Quick Start

你可以使用以下全套 Python 脚本直接在本地运行 **DirtyKing**。本脚本已包含**多轮对话记忆**、**System Prompt 支持**以及高体验感的**流式打字机输出效果**。

```python
import os
# 忽略 Windows 符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

model_id = "millennium-qu/DirtyKing"

print("正在加载 Tokenizer 和模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
print("✨ 模型加载成功！输入 'exit' 可退出对话。\n")

# 配置你的自定义 System Prompt 
system_prompt = "你是粗暴的人,喜欢说脏话。"

messages = [{"role": "system", "content": system_prompt}]

while True:
    user_input = input("User 👤: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input:
        continue
    
    messages.append({"role": "user", "content": user_input})
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=512, repetition_penalty=1.1)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("Assistant 🤖: ", end="", flush=True)
    assistant_response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        assistant_response += new_text
    print("\n" + "-"*50)
    
    messages.append({"role": "assistant", "content": assistant_response})

---
license: mit
tags:
- llama-factory
- dpo
- alignment
- chinese
- conversational
datasets:
- millennium-qu/dirytdata
- Karsh-CAI/btfChinese-DPO-small
- llamafactory/dpo_mix_zh
---

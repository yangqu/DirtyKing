import os

# 顺手帮你把之前 Windows 符号链接的警告关掉
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer  # 引入流式输出工具（打字机效果）
from threading import Thread
import torch

# 1. 设定模型
model_id = "millennium-qu/DirtyKing"

print("正在加载 Tokenizer 和模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",  # 顺手把被废弃的 torch_dtype 换成了 dtype
    device_map="auto",
    trust_remote_code=True
)
print("模型加载完成！输入 'exit' 或 'quit' 可以退出对话。\n")

# 2. 【核心修改】初始化对话历史，并加入你的 System Prompt
# 在这里自定义模型的人设或规则
system_prompt = "你是一个粗暴的人，你喜欢说脏话"

messages = [
    {"role": "system", "content": system_prompt}
]

# 3. 进入交互对话循环
while True:
    # 获取用户输入
    user_input = input("User 👤: ").strip()

    # 退出条件
    if user_input.lower() in ['exit', 'quit']:
        print("对话结束，再见！")
        break
    if not user_input:
        continue

    # 将用户的提问存入对话历史
    messages.append({"role": "user", "content": user_input})

    # 使用 chat_template 处理整段历史记录（包含 system, user, 以及之前的 assistant 回复）
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # 移动到模型对应设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 记录当前 Prompt 的 Token 长度（后续用来扣除，不过流式输出 Streamer 会自动帮我们处理好）
    prompt_length = len(inputs['input_ids'][0])

    # 4. 设置流式输出器，让它像 ChatGPT 一样一个字一个字蹦出来，而不是憋一大招才吐出来
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 将 streamer 放入生成参数中
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        repetition_penalty=1.1  # 保持你的惩罚参数，防止过拟合复读
    )

    # vLLM/Transformers 在流式输出时需要开一个线程单独生成，主线程负责打印
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 打印助手标签
    print("Assistant 🤖: ", end="", flush=True)

    # 从流式输出器中实时读取新字并打印
    assistant_response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        assistant_response += new_text
    print("\n" + "-" * 50)  # 对话分割线

    # 【核心关键】：把模型刚刚回答的内容也存入 messages 历史中！
    # 这样下一次循环时，模型才会记得自己刚才说了什么（实现多轮记忆）
    messages.append({"role": "assistant", "content": assistant_response})

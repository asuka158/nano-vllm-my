import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 1. 选模型
    path = os.path.expanduser("~/models/Qwen3-0.6B/")
    # 2. 准备tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 3. 初始化高层推理入口
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 4. 准备本次请求的采样参数
    # 对于每个输出，计算得到的是不同token的概率分布，并不是一定输出概率最高的token
    # 而是根据SamplingParams进行采样
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    
    # 5. 准备并格式化 prompt
    prompts = [
        "介绍一下你自己",
        "你好",
    ]
    # for i, prompt in enumerate(prompts):
    #     encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    #     input_ids = encoded["input_ids"]

    #     print(f"\n===== Prompt {i} =====")
    #     print("prompt string:")
    #     print(repr(prompt))

    #     print("input_ids shape:")
    #     print(input_ids.shape)   # [1, seq_len]

    #     print("input_ids values:")
    #     print(input_ids)

    #     print("tokens:")
    #     print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    #     print(tokenizer.decode(input_ids[0][0]))
    # 只是给prompts加前缀后缀之类的，不做字符串->tokenid的映射，所以传给llm.generate的还是字符串
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 6. 调用 llm.generate(...) 拿回结果
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()

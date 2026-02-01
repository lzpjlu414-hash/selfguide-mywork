
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# baseline.py 的作用是：
# 使用 LLaMA 模型对测试集进行 batch 推理，
# 将模型生成的答案逐条保存为 JSONL 文件，
# 并通过字符串匹配的方式计算预测准确率。


import fire
import json
from llama import Llama
import pandas as pd
from tqdm import tqdm

def main(
    ckpt_dir: str,  #LLaMA 权重目录
    tokenizer_path: str,  #tokenizer.model 路径
    dataset: str,  #数据集名（如 date）
    temperature: float = 0.2, #采样温度（越低越稳定）
    top_p: float = 0.9,  #nucleus sampling 每一步生成时，只在“累计概率达到 p 的那一小撮最可能的词”里随机选一个
    max_seq_len: int = 1024,  #输入最大长度
    max_gen_len: int = 128,   #生成长度
    max_batch_size: int = 8,  #batch 推理大小
):
    generator = Llama.build(  #构建 LLaMA 推理器
        ckpt_dir=ckpt_dir,                    #加载模型权重
        tokenizer_path=tokenizer_path,        #加载 tokenizer
        max_seq_len=max_seq_len,              #初始化 KV cache
        max_batch_size=max_batch_size,
    )
    #初始化缓存 & 读取数据
    new_rows_list=[]        #临时存 一个 batch 的结果
    data = pd.read_json(f"data/{dataset}/test.jsonl", lines=True)    #{"id":1,"question":"...","answers":["..."]}
    output = f"log/{dataset}.jsonl"
    print(output)

    with open(output, 'w', encoding='utf-8') as file:  #清空输出文件  确保每次运行都是 全新结果
        file.write('') 

    #构造 prompt（非常关键）  模型 只看到 input
    data['input']=""
    for i in range(len(data)):
        data.loc[data.index[i],'input'] =  "Question:" + data['question'][i] + "\n" "Answer:"

    #Batch 推理（核心循环）
    for i in tqdm(range(0, len(data), max_batch_size)):
        batch = data.iloc[i:i+max_batch_size]

        #调用 LLaMA 生成
        results = generator.text_completion(
            batch['input'],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # 收集生成结果
        for index, result in zip(batch.to_dict(orient='records'), results):
            new_rows_list.append({
                'id': index['id'],
                'baseline_answer': result['generation'],
                'ground_truth': index["answers"]  #标准答案（通常是列表）
            })  #{"id":12,"baseline_answer":"1999","ground_truth":["1999"]}



        # Write the current batch to the JSON Lines file
        # with open(output, 'a', encoding='utf-8') as jsonl_file:
        #     for row in new_rows_list:
        #         jsonl_file.write(json.dumps(row) + '\n')
        #         new_rows_list=[]
        with open(output, 'a', encoding='utf-8') as jsonl_file:
            for row in new_rows_list:
                jsonl_file.write(json.dumps(row) + '\n')

        new_rows_list = []

    n = 0  #总样本数
    x = 0  #命中数
    # Read JSON Lines file and create a DataFrame
    with open(output, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                n+=1  #统计样本总数
                data = json.loads(line)   #把字符串变成 dict
                ground_truth = data['ground_truth']
                answers = str(data['baseline_answer'])
                if any(answer.lower() in answers.lower() for answer in ground_truth): #只要模型输出中包含任意一个标准答案字符串 → 判为正确
                    x+=1  #命中数 +1

    print(x/n) #print(x/n)


if __name__ == "__main__":   #主程序才执行
    fire.Fire(main)          #解析命令行参数

"""
python baseline.py \
    --ckpt_dir /data/LLaMA-2/meta/Llama-2-7b-meta \
    --tokenizer_path /data/LLaMA-2/meta/Llama-2-7b-meta/tokenizer.model  \
    --dataset date --max_seq_len 1024 --max_batch_size 8
"""

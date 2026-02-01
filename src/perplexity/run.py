import argparse
import torch
from tqdm import tqdm
#import models #你项目里的自定义模块
from src.perplexity.llama_model import create_model as models
import utils
import json
import pandas as pd
import numpy as np



#从一个 jsonl 数据文件里，逐条取出 answer 文本，让模型算出这些文本的 token-level logprob（对数概率），并把所有 logprob 拼成一个长数组返回。
# def compute_perplexity_data(model, data_path):
#     # For expedience, we're going to assume everything fits in memory for now
#     # Also for expedience we're just going to save lists of arrays
#     probs= np.array([]) #初始化一个空数组 用来存放 logprob
#     data = pd.read_json(data_path, lines=True)
#     for i in tqdm(range(len(data))):
#         output = model.get_perplexity_data(data["answer"][i])  #取第 i 条的 answer 文本并请求模型算 logprob
#         probs = np.append(probs, output['logprobs']) #把当前文本的 logprobs 拼到总数组里
#     return probs
def compute_perplexity_data(model, data_path):
    """
    更高效的版本：用 list 收集每条样本的 logprobs，最后 np.concatenate 一次性拼接。
    """
    data = pd.read_json(data_path, lines=True)

    logprobs_list = []  # 用列表收集每条样本的 logprobs（避免循环中反复 np.append）
    for i in tqdm(range(len(data))):
        answer_text = data["answer"][i]
        output = model.get_perplexity_data(answer_text)
        logprobs_list.append(np.asarray(output["logprobs"]))

    # 如果数据为空，返回空数组，避免 concatenate 报错
    if not logprobs_list:
        return np.array([])

    return np.concatenate(logprobs_list)



def main():
    model = models.create_model() #它会返回一个模型对象，里面必须实现 get_perplexity_data()
    #data_path = "/data/llama-meta/data/data.jsonl"
    data_path = "data/CLUTRR/test.jsonl"  # 改成你的真实路径

    perplexity_data = compute_perplexity_data(  #计算 logprobs array([-0.1, -1.4, -0.7, ...])
        model=model,
        data_path=data_path,
    )


    aggregate_logprobs = perplexity_data #这句本身没做任何处理，就是换了个名字
    perplexity = np.exp(-aggregate_logprobs.mean())  #计算困惑度 perplexity
    result = {
        "perplexity": float(perplexity)  #组织输出结果
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
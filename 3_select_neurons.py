import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import Trace, TraceDict
import argparse
import jsonlines
import logging
import numpy as np
import os
import json
import random
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import model_name2path, get_interveted_output
import torch.nn.functional as F
from functools import partial
import json
import numpy as np
from collections import Counter

# file_suffix = {
#     'llama7b': 'llama7b',
#     'gpt2-small':'small',
#     'gpt2-xl':'xl',
#     'gpt-neo':'neo',
# }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path):
    attrs = []
    with jsonlines.open(data_path, 'r') as fr:
        for line in fr:
            if 'attributuon_scores' in line.keys():
                attrs.append(line)

    #     line = f.readline().strip()
    #     attr = json.loads(line)
    
    return attrs

def get_pn(res, threshold):
    total_counts = Counter()

    for sublist in res:
        sublist_counts = Counter(set(sublist))
        total_counts.update(sublist_counts)

    threshold_frequency = max(1, threshold * len(res))

    frequent_elements = [element for element, count in total_counts.items() if count > threshold_frequency]

    return frequent_elements

def filter(attr, threshold):
    attr_array = np.array(attr)

    max_abs_value = np.max(np.abs(attr_array))
    threshold = threshold * max_abs_value

    res = []
    for i in range(attr_array.shape[0]):
        for j in range(attr_array.shape[1]):
            temp = {}
            if abs(attr_array[i, j]) > threshold:
                # temp[f"{i}:{j}"] = attr_array[i, j]
                # res.append(temp)
                res.append(f"{i}:{j}")

    return res

def rand_sample(attr_scores, sample_num):
    # print(type(attr_scores))
    # print(attr_scores[0]['prompt'])
    # print(attr_scores[0]['privacy'])
    sampled_scores = random.sample(attr_scores, sample_num)
    privacys = []
    for i in sampled_scores:
        privacys.append(i['prompt']+"#####"+i['privacy'])

    return sampled_scores, privacys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['llama7b', 'gpt2', 'gpt2-xl', 'gpt-neo', 'qwen2.5'],
                        )
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_type', type=str, default=None)
    parser.add_argument('--text_threshold', type=float, default=0.1)
    parser.add_argument('--batch_threshold', type=float, default=0.5)

    args = parser.parse_args()

    attr_scores = read_data(args.data_path)
    logger.info(f"the number of memorized privacy: {len(attr_scores)}")

    if args.sample_num:
        attr_scores, privacys = rand_sample(attr_scores, args.sample_num)

    all_res = []
    ## 筛选每个sample的神经元
    for attr in attr_scores:
        res = filter(attr['attributuon_scores'], args.text_threshold)
        all_res.append(res)
    
    pn = get_pn(all_res, args.batch_threshold)
    logger.info(f"the number of privacy neurons: {len(pn)}")


    # Split the data into individual neurons and parse them
    layers, positions = zip(*[map(int, neuron.split(':')) for neuron in pn])

    layers_np = np.array(layers)

    # Determine the unique number of layers to define the y-axis
    unique_layers = np.unique(layers_np)

    # Count the number of neurons in each layer
    neuron_counts_per_layer = {layer: np.sum(layers_np == layer) for layer in unique_layers}

    if args.sample_num:
        output_pn_dir = f'./pn_result/sampled-filtered-pn-{args.model_name}-{args.data_type}-s{args.sample_num}.txt'
        with open(os.path.join(output_pn_dir), 'w') as fw:
            for i in pn:
                fw.write(i+'\n')
        output_priv_dir = f'./pn_result/sampled-{args.model_name}-{args.data_type}-memorizations-s{args.sample_num}.txt'
        with open(os.path.join(output_priv_dir), 'w') as fw:
            for i in privacys:
                fw.write(i+'\n')
    else:
        output_pn_dir = f'./pn_result/filtered-pn-{args.model_name}-{args.data_type}-all.txt'
        with open(os.path.join(output_pn_dir), 'w') as fw:
            for i in pn:
                fw.write(i+'\n')




if __name__ == "__main__":
    main()
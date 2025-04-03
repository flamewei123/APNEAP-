import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from baukit import Trace, TraceDict
import argparse
import jsonlines
import logging
import numpy as np
import os
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import model_name2path, get_interveted_output, get_layer_names
import torch.nn.functional as F
from functools import partial
import re

file_suffix = {
    'llama7b': 'llama7b',
    'gpt2-small':'small',
    'gpt2-xl':'xl',
    'gpt-neo':'neo',
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_attributions(model, prompts, tokenizer, gold_labels, device=0, model_name='llama7b', num_batch=10):

    # only focus on the medium activations of mlp
    HEADS, MLPS, LAYERS = get_layer_names(model_name, model)

    targets = MLPS #LAYERS
    model.eval()
    prompts = prompts.to(device) 
    gold_labels = gold_labels.to(device)
    
    with TraceDict(model, targets, retain_grad=True) as ret:
        output = model(prompts, labels=gold_labels)

        # predicted_ids = output.logits.argmax(dim=-1).squeeze().tolist()
        # decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_ids]
        # # 打印解码结果
        # for i, decoded_output in enumerate(decoded_outputs):
        #     print(f"解码后的输出结果 {i}: {decoded_output}")

        target_probs, target_pos_list = cal_prob(output.logits, gold_labels)
    
    mlp_wise_hidden_states = [ret[t].output.squeeze() for t in MLPS]
    # print(mlp_wise_hidden_states[0].size()) # torch.Size([m, seq_lgth, hdsz])
    logger.info(f"original hidden states have been extracted")


    ffn_attr_dict = []
    for idx, layer_name in enumerate(MLPS):

        weights_step_list, scaled_weights_list = get_ori_ffn_weights(mlp_wise_hidden_states[idx], target_pos_list, num_batch)  # []*batch-szie
        # print(weights_step_list)

        batch_grads_list = []
        ## calculate privacy attribution by changing neuron activations
        for i in range(num_batch): # 替换m次
            intervene_fn = partial(intervene_fn_replace, tar_pos = target_pos_list, repl_states = [scaled_weights_list[j][i] for j in range(len(scaled_weights_list))])
            # 用scale的激活值替换原值，找到target position，会被替换的有[m,1,hdsz]
            with Trace(model, layer_name, edit_output=intervene_fn, retain_grad=True) as retl: 
                output = model(prompts, labels=gold_labels, output_hidden_states = True)
                loss = output.loss
                loss.backward(retain_graph=True)
                target_probs, target_pos_list = cal_prob(output.logits, gold_labels)
            mlp_hidden_state_grad = retl.output.grad # [m,seq_lgth,hdsz]

            # 清理不需要的计算图和梯度信息
            del loss, retl.output.grad
            torch.cuda.empty_cache()
            
            # 避免概率相除导致nan和inf
            if model_name == 'llama7b':
                mlp_hidden_state_grad = mlp_hidden_state_grad.float()
            
            
            ## text_batch上的梯度乘以输出概率的倒数
            new_hidden_grads = []
            for i in range(len(target_pos_list)):
                target_hidden_grads = mlp_hidden_state_grad[i, int(target_pos_list[i])-1, :] / target_probs[i]
                new_hidden_grads.append(target_hidden_grads)           
            
            ## 将处理后的隐状态堆叠成一个新的张量
            stacked_hidden_grads = torch.stack(new_hidden_grads)            
            weighted_ori_hidden_states = torch.stack(weights_step_list)
        
            new_hidden_grads = torch.mul(weighted_ori_hidden_states, stacked_hidden_grads) # [m,hdsz]

            ## 沿着text-batch维度(第0维)求和，以得到累加后的结果
            sequence_grads = torch.sum(new_hidden_grads, dim=0)

            batch_grads_list.append(sequence_grads)

        ## 沿着num_batch维度(第0维)求和，以得到梯度累加后的结果
        stack_batch_grads = torch.stack(batch_grads_list)
        num_batch_grads = torch.sum(stack_batch_grads, dim=0)
        
        ffn_attr_dict.append(num_batch_grads.tolist()) 

    return ffn_attr_dict

def intervene_fn_replace(original_state, layer_name, tar_pos, repl_states):
    '''
    original_state： hidden states of specifuc layer
    layer_name: module name of mlp layer
    
    '''
    for idx, repl_state in enumerate(repl_states):
        original_state[idx,int(tar_pos[idx]),:] = repl_state

    return original_state

def read_priv_data(data_path):
    datas = []
    with open(data_path,"r+")as f:
        lines = f.readlines()
        for line in lines:
            priv_text = line.split('#####')[0]
            privacy = line.split('#####')[1]
            datas.append([priv_text,privacy.strip()])

    return datas

def cal_prob(logits, gold_labels):
    # 计算目标token的概率
    target_probs = []
    target_pos = []

    for idx, (logit, label) in enumerate(zip(logits, gold_labels)):
        # 找到非-100的标签的索引
        target_index = (label != -100).nonzero(as_tuple=True)[0]
        # 如果存在非-100的标签
        if len(target_index) > 0:
            # 仅获取第一个非-100标签的索引（假设每个样本只有一个目标）
            target_index = target_index[0]
            target_pos.append(target_index)
            # 计算softmax
            softmax_probs = F.softmax(logit[target_index], dim=-1)
            # 提取目标概率
            target_prob = softmax_probs[label[target_index]].item()
            target_probs.append(target_prob)
        else:
            # 如果没有目标标签，可以添加一个默认值，例如0或None
            target_probs.append(0)
    
    return target_probs, target_pos

def scale_hidden_states(neuron_activation, num_batch):
    # print(neuron_activation.size()) # [hdsz]
    baseline = torch.zeros_like(neuron_activation)
    step = (neuron_activation - baseline) / num_batch

    res = [torch.add(baseline, step * i) for i in range(num_batch)]  # num_batch*[ffn_size]

    return res, step

def get_ori_ffn_weights(mlp_wise_hidden_states, target_pos_list, num_batch=10):
    weights_step_list = []
    scaled_weights_list = []

    ## get original neuron activations, and scale activations for Riemann approximation
    for priv_idx in range(len(mlp_wise_hidden_states)):
        target_pos = target_pos_list[priv_idx]
        neuron_activation = mlp_wise_hidden_states[priv_idx:priv_idx+1, target_pos:target_pos+1, :].squeeze() 
        # print(neuron_activation.size()) # [hdsz]

        scaled_weights, weights_step = scale_hidden_states(neuron_activation, num_batch) 
        weights_step_list.append(weights_step) # 原始激活值除以m， batch-size*[hdsz]
        scaled_weights_list.append(scaled_weights) # 递增的激活值， batch-size*[num_batch*[hdsz]]
    return weights_step_list, scaled_weights_list

def find_sublist_index(a, b):
    len_a = len(a)
    len_b = len(b)
    
    # 遍历a，寻找b的起始位置
    for i in range(len_a - len_b + 1):
        if a[i:i + len_b] == b:
            return i
    
    return None

def TextToBatch(input_text, priv_text, tokenizer, model_name, max_length = 128):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if input_text[input_text.find(priv_text)-1] == ' ':
        priv_text = ' ' + priv_text

    # 准备输入和标签
    input = tokenizer(input_text,return_tensors='pt',truncation=True)
    priv = tokenizer(priv_text,return_tensors='pt',truncation=True)
    
    if model_name == 'llama7b':
        input_tokens = tokenizer.convert_ids_to_tokens(input['input_ids'][0][1:])
        priv_tokens = tokenizer.convert_ids_to_tokens(priv['input_ids'][0][2:])
        first_index = find_sublist_index(input_tokens,priv_tokens)+1
    else:
        input_tokens = tokenizer.convert_ids_to_tokens(input['input_ids'][0])
        priv_tokens = tokenizer.convert_ids_to_tokens(priv['input_ids'][0])
        first_index = find_sublist_index(input_tokens,priv_tokens)

    if first_index:
        end_index = first_index + len(priv_tokens)

        # 对所有输入进行编码并应用填充
        input_ids = [input['input_ids'][0][:first_index + idx] for idx in range(len(priv_tokens))]

        padded_input_ids = torch.stack([torch.nn.functional.pad(input_id, (0, max_length - len(input_id)), value=tokenizer.pad_token_id) for input_id in input_ids])
        # print(padded_input_ids[0,:])

        padded_label_ids = [[-100 for i in range(max_length)] for i in range(len(priv_tokens))] # 使用 -100 初始化
        for i, label_id in enumerate(input['input_ids'][0][first_index:end_index].tolist()):
            padded_label_ids[i][len(input_ids[i])] = label_id
        padded_label_ids = torch.tensor(padded_label_ids)
        # print(padded_label_ids[0,:])

        return padded_input_ids, padded_label_ids
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['llama7b', 'gpt2', 'gpt2-xl', 'gpt-neo', 'qwen2.5'],
                        )
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--privacy_kind', type=str, default='TEL')
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--num_batch', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--use_fp16', action='store_true', help="Enable fp16 precision")
    args = parser.parse_args()

    # set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.device) == 1:
        device = torch.device("cuda:%s" % args.device)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    dtype = torch.float16 if args.use_fp16 else torch.float32
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    
    # load target private data
    datas = read_priv_data(args.data_path)

    logger.info(f"the number of memorized privacy: {len(datas)}")

    # all_attr_results = []

    for data in tqdm(datas):
        
        priv_text, privacy = data[0], data[1]
        result_dict = {}

        prompt, gold_labels = TextToBatch(priv_text, privacy, tokenizer, args.model_name, args.max_length)
        if prompt is not None:
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                MLP_attribution_scores_list = get_attributions(model, prompt, tokenizer, gold_labels, device, args.model_name, args.num_batch)
           
            result_dict['prompt'] = priv_text
            result_dict['privacy'] = privacy
            result_dict['attributuon_scores'] = MLP_attribution_scores_list
        # 保存结果    
        output_dir = f'./pn_result/{args.model_name}_{args.privacy_kind}.jsonl'
        with jsonlines.open(os.path.join(output_dir), 'a') as fw:
            fw.write(result_dict)
    logger.info(f"attribution scores have been calculated")

    


        
        

 
if __name__ == "__main__":
    main()
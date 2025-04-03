import logging
import argparse
import math
import torch
import re
import random
import numpy as np
import random
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from functools import partial
from baukit import Trace, TraceDict
import matplotlib.pyplot as plt


import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    default_data_collator,
)
import torch.nn.functional as F
from utils import get_candidate_priv_encode, get_tar_rank, get_exposure, get_layer_names, exact_match, beam_match

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
TOTAL_CANDIDATES = 10_000_000_000

def eval_ppl(eval_dataloader,model,device,batch_size,editing_kind=None,MLPS=None,pn_dict=None,vector=None):
    losses = []
    if editing_kind is None:
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            loss = outputs.loss
            losses.append(loss.repeat(batch_size))
    if editing_kind == "zero":
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intervene_fn = partial(intervene_fn_zero, pn_dict = pn_dict)
                with TraceDict(model, MLPS, edit_output=intervene_fn) as ret: 
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            loss = outputs.loss
            losses.append(loss.repeat(batch_size))
    if editing_kind == "patching":
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intervene_fn = partial(intervene_fn_add_all, pn_dict = pn_dict, vector=vector)
                with TraceDict(model, MLPS, edit_output=intervene_fn) as ret: 
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            loss = outputs.loss
            losses.append(loss.repeat(batch_size))


    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf") #

    logger.info(f" perplexity: {perplexity*18.143/42.166} eval_loss: {eval_loss}")
    return perplexity


def read_priv_data(data_path):
    datas = []
    with open(data_path,"r+")as f:
        lines = f.readlines()
        for line in lines:
            priv_text = line.split('#####')[0]
            privacy = line.split('#####')[1]
            datas.append([priv_text,privacy.strip()])

    return datas


def get_text_batch(prompt, secret):
    scaled_input_texts = []
    for i in range(len(secret)):
        sliced_secret = ' '.join(list(secret)[:i+1])
        scaled_input_texts.append(prompt+" "+sliced_secret)
    return scaled_input_texts

def load_pn(privacy_neuron_path, n_layer, MLPS):
    with open(privacy_neuron_path,'r') as file:
        lines = file.readlines()
        pn_dict = {}

        for i in lines:
            temp = i.strip().split(':')
            layer_name = MLPS[int(temp[0])]
            if layer_name not in pn_dict:
                pn_dict[layer_name] = [int(temp[1])]
            else:
                pn_dict[layer_name].append(int(temp[1]))
    return pn_dict

def get_patching_vector(privacys, model, tokenizer, MLPS, device, model_name, harmless_words):
    model.eval()
    all_hidden_states = []
    for privacy in privacys:
        secret = privacy[1]
        harmless = privacy[0].replace(secret,' '.join(list(harmless_words)))

        input_txt = [privacy[0],harmless]
        inputs = tokenizer(input_txt, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)

        targets = MLPS 
        with torch.no_grad():
            with TraceDict(model, targets) as ret:
                output = model(**inputs, output_hidden_states = True)
            mlp_wise_hidden_states = [ret[t].output.squeeze().detach().cpu() for t in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze() # [layer_n,2,length,hidden-szie]
            differ_vector = mlp_wise_hidden_states[:, 0, :, :] - mlp_wise_hidden_states[:, 1, :, :]
            all_hidden_states.append(torch.mean(differ_vector[:,:,:],dim=1))
    stacked_vectors = torch.stack(all_hidden_states, dim=0)
    steer_vector = torch.mean(stacked_vectors, dim=0)

    return steer_vector

def intervene_fn_zero(original_state, layer, pn_dict):
    '''
    layer_name: module name of medium mlp layer
    pn_dict: position of privacy neurons  # pn_dict['transformer.h.11.mlp.c_fc'] = [15,1232,...,3070]
    '''  
    layers_to_intervene = pn_dict.keys()
    if layer in layers_to_intervene:
        for position in pn_dict[layer]:
            original_state[:,:,position] = 0
    return original_state


def intervene_fn_add(original_state, layer, pn_dict, vector, alpha=1):
    '''
    layer_name: module name of medium mlp layer
    pn_dict: position of privacy neurons  # pn_dict['transformer.h.11.mlp.c_fc'] = [15,1232,...,3070]
    vector: steer vector for harmless words # list:[layer_n, hidden_size]
    alpha: scalar, strength of intervention
    '''  
    layers_to_intervene = pn_dict.keys()
    if layer in layers_to_intervene:
        for position in pn_dict[layer]:
            original_state[:,:,position] += alpha * vector[int(layer.split('.')[2]),position]
    return original_state

def intervene_fn_add_all(original_state, layer, pn_dict, vector, alpha=1):
    if int(layer.split('.')[2]) in [15]:
        for position in range(vector.shape[1]):
            original_state[:,:,position] += alpha * vector[int(layer.split('.')[2]),position]
    return original_state

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Whole private data path. ")
    parser.add_argument("--validation_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="validation data path for performance observation. ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--privacy_neuron_path", default=None, type=str, required=True)
    parser.add_argument("--editing_kind", default=None, type=str)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument("--steer_words",
                        type=str,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=64)
    
    # parse arguments
    args = parser.parse_args()
    text_file = args.validation_data_path  
    max_length = args.max_seq_length  
    batch_size = args.batch_size

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load LM
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    _, MLPS, _ = get_layer_names(args.model_name, model)
    model.to(device)
    model.eval()

    # load validation data
    raw_datasets = load_dataset('text', data_files=text_file)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if max_length > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({max_length}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(max_length, tokenizer.model_max_length)
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    
    eval_dataloader = DataLoader(
        lm_datasets['train'], collate_fn=default_data_collator, batch_size=batch_size
    )
    # load private data
    unique_privacys = read_priv_data(args.priv_data_path)
    # nums_encode = get_candidate_priv_encode(tokenizer,unique_privacys)

    if not args.editing_kind:
        # # ======================== eval model on valid =================================     
        logger.info(f"start evaluating original target model")
        eval_ppl(eval_dataloader,model,device,batch_size)

        # ======================== eval model leakage =================================
        # exp_sum = 0
        # count = 0   
        # exposure_list = []
        em_sum = 0

        for privacy in unique_privacys:  
            input_text, priv_text = privacy[0], privacy[1]

            if input_text[input_text.find(priv_text)-1] == ' ':
                priv_text = ' ' + privacy[1]

            inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
            inputs.to(device)

            outputs = model(**inputs)
            em = exact_match(priv_text,outputs,tokenizer)

            # em = beam_match(priv_text, model, inputs, tokenizer, max_length, num_return_sequences=1)
            
            em_sum += em

        logger.info(f"***** total em: { em_sum } *****")

    if args.editing_kind:
        ### ======================== eval after model editing =================================
        # load privacy neurons
        logger.info(f"start editing target model")
        if args.model_name == 'gpt2':
            pn_dict = load_pn(args.privacy_neuron_path, config.n_layer, MLPS)
        if args.model_name == 'gpt-neo': 
            pn_dict = load_pn(args.privacy_neuron_path, config.num_layers, MLPS)
        if args.model_name == 'qwen2.5' or 'llama7b': 
            pn_dict = load_pn(args.privacy_neuron_path, config.num_hidden_layers, MLPS)
        
        if args.editing_kind == "patching": # get steering vector
            # all_privacys = load_privacys(args.priv_data_path, 1)
            steering_vector = get_patching_vector(unique_privacys, model, tokenizer, MLPS, device, args.model_name, args.steer_words)

        # # ======================== eval model on valid =================================
        if args.editing_kind == 'patching':     
            eval_ppl(eval_dataloader,model,device,batch_size,args.editing_kind,MLPS,pn_dict,steering_vector)
        if args.editing_kind == 'zero':
            eval_ppl(eval_dataloader,model,device,batch_size,args.editing_kind,MLPS,pn_dict)

        # # ======================== eval model leakage ================================= 
        # exp_sum = 0   
        # count = 0
        # new_result = []
        em_sum = 0
        for privacy in unique_privacys:  
            input_text, priv_text = privacy[0], privacy[1]

            if input_text[input_text.find(priv_text)-1] == ' ':
                priv_text = ' ' + privacy[1]

            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)

            if args.editing_kind == "zero":
                intervene_fn = partial(intervene_fn_zero, pn_dict = pn_dict)
                with TraceDict(model, MLPS, edit_output=intervene_fn) as ret:
                    outputs = model(**inputs)
                em = exact_match(priv_text,outputs,tokenizer)
                em_sum += em
            
            if args.editing_kind == "patching":
                intervene_fn = partial(intervene_fn_add_all, pn_dict = pn_dict, vector=steering_vector, alpha=10)
                with TraceDict(model, MLPS, edit_output=intervene_fn) as ret:
                    outputs = model(**inputs)
                em = exact_match(priv_text,outputs,tokenizer)
                em_sum += em


        

        logger.info(f"***** total em: { em_sum } *****")






if __name__ == "__main__":
    main()
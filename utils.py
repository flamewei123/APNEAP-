from baukit import Trace, TraceDict
from einops import rearrange
from functools import partial
import torch
import math
from torch.nn.utils.rnn import pad_sequence
from itertools import chain

# layer_name_temp = {
#     'llama7b':"model.layers.{}.mlp.up_proj",
#     'gpt2-small':"transformer.h.{}.mlp.c_fc",
#     'gpt2-xl':"transformer.h.{}.mlp.c_fc",
#     'gpt-neo':"transformer.h.{}.mlp.c_fc",
# }

layer_name_temp = {
    'llama7b':"model.layers.{}",
    'gpt2-small':"transformer.h.{}",
    'gpt2-xl':"transformer.h.{}",
    'gpt-neo':"transformer.h.{}",
}

model_name2path = {
    'llama7b': '/data1/willow/models/llama7b-hf',
    'gpt2-small':'/data1/willow/models/gpt2',
    'gpt2-xl':'/data1/willow/models/gpt2-xl',
    'gpt-neo': '/data1/willow/models/gpt-neo-2.7B',
    # 'gpt2-small':'/data1/willow/models/medical_gpt/ep9',
    # 'gpt2-xl':'/data1/wuxinwei/model/0926_medical_gpt2xl',
    # 'gpt-neo':'/data1/willow/models/medical_gpt/neo_ep5',
}

file_suffix = {
    'llama7b': 'llama7b',
    'gpt2-small':'small',
    'gpt2-xl':'xl',
    'gpt-neo':'neo',
}

def get_candidate_priv_encode(tokenizer,privacys):
    priv_encode={}
    len_count = 0
    # privacys = privacys[:10]
    for privacy in privacys:
        input_text, priv_text = privacy[0], privacy[1]
        if input_text[input_text.find(priv_text)-1] == ' ':
            priv_text = ' ' + privacy[1]
        input = tokenizer(input_text,return_tensors='pt',padding=True,truncation=True)
        priv = tokenizer(priv_text,return_tensors='pt',padding=True,truncation=True)
        # print(tokenizer.convert_ids_to_tokens(input['input_ids'][0]))
        encodes = priv['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(priv['input_ids'][0])
        len_count += len(tokens)
        if len(encodes) == len(tokens):
            for i in range(len(encodes)):
                if tokens[i] not in priv_encode:
                    priv_encode[tokens[i]] = encodes[i]
        else:
            print("#### ",priv_text)
    # print('average privacy length:',len_count/len(privacys))
    # print('candidate priv token:', len(priv_encode))
    return priv_encode

def get_layer_names(model_name, model):
    if model_name == 'llama7b':
        HEADS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp.up_proj" for i in range(model.config.num_hidden_layers)]
        LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
        
    elif 'gpt2' in model_name:
        HEADS = [f"transformer.h.{i}.attn" for i in range(model.config.n_layer)]
        MLPS = [f"transformer.h.{i}.mlp.c_fc" for i in range(model.config.n_layer)]
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.n_layer)]

    elif 'gpt-neo' in model_name:
        HEADS = [f"transformer.h.{i}.attn" for i in range(model.config.num_layers)]
        MLPS = [f"transformer.h.{i}.mlp.c_fc" for i in range(model.config.num_layers)]
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.num_layers)]
    
    elif 'qwen2.5' in model_name:
        HEADS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp.up_proj" for i in range(model.config.num_hidden_layers)]
        LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]

    
    return HEADS, MLPS, LAYERS

# def exact_match(secret,outputs,tokenizer):
#     result = tokenizer.decode(torch.argmax(outputs['logits'], dim=-1)[0], skip_special_tokens=True)
#     if secret in result:
#         return 1
#     else:
#         return 0
def exact_match(secret, outputs, tokenizer):

    result = tokenizer.decode(torch.argmax(outputs['logits'], dim=-1)[0], skip_special_tokens=True)
    if secret in result:
        return 1
    else:
        return 0

def beam_match(secret, model, inputs, tokenizer, max_length, num_return_sequences=50):
    outputs = model.generate(
        **inputs,
        num_beams=num_return_sequences,
        num_return_sequences=num_return_sequences,
        max_length=max_length,  # 根据需要调整最大长度
        early_stopping=True
    )
    print(len(outputs))
    for seq in outputs:
        result = tokenizer.decode(seq, skip_special_tokens=True)
        print(result)
        if secret in result:
            return 1
    return 0





def find_sublist_index(a, b):
    len_a = len(a)
    len_b = len(b)
    
    # 遍历a，寻找b的起始位置
    for i in range(len_a - len_b + 1):
        if a[i:i + len_b] == b:
            return i
    
    return None

def get_exposure(input_text,priv_text,outputs,nums_encode,tokenizer):
    rank = 1

    # get first and end index of privacy
    input = tokenizer(input_text,return_tensors='pt',padding=True,truncation=True)
    priv = tokenizer(priv_text,return_tensors='pt',padding=True,truncation=True)
    input_tokens = tokenizer.convert_ids_to_tokens(input['input_ids'][0])
    priv_tokens = tokenizer.convert_ids_to_tokens(priv['input_ids'][0])
    first_index = find_sublist_index(input_tokens,priv_tokens)

    if first_index:
        end_index = first_index + len(priv_tokens)
        for i in range(first_index,end_index):
            tar_rank = get_tar_rank(input_tokens[i],outputs.logits[0][i],nums_encode)*pow(len(nums_encode),len(priv_tokens)-1-i+first_index)
            rank += tar_rank
        
        canary_exposure = math.log(pow(len(nums_encode),len(priv_tokens)), 2) - math.log(rank, 2)

        return rank, canary_exposure
    else:
        return None, None

def get_tar_rank(target,output,nums_encode):
    values = {}
    for i in nums_encode.keys():
        values[i] = float(output[nums_encode[i]])
    ranked_vals = dict(sorted(values.items(), key=lambda x: x[1],reverse = True))  
    rank = list(ranked_vals.keys()).index(target)
    return rank


def group_texts(examples, block_size):
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

def eval_interved_ppl(eval_dataloader,model,device,batch_size,args,window_ids):
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}
            outputs = get_interveted_output(model,inputs,model_name=args.model_name, window_size=args.window_size,
                                        window_ids=window_ids,alpha=args.steering_strength,
                                        intervene_method=args.edit_type, steer_vector_path=args.steering_vector_path)

        loss = outputs.loss
        losses.append(loss.repeat(batch_size))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f" perplexity: {perplexity} eval_loss: {eval_loss}")
    return perplexity

def intervene_fn_zero(original_state, layer, window_size, window_ids):
    '''
    layer_name: module name of medium mlp layer
    window_size: size of a window to intervene
    window_ids: dict of {layer_name: [window_ids]}
    '''
    layers_to_intervene = window_ids.keys()
    if layer in layers_to_intervene:
        for window_id in window_ids[layer]:
            original_state[:,:,window_id*window_size:(window_id+1)*window_size] = 0
    return original_state

def intervene_fn_neg(original_state, layer, window_size, window_ids):
    '''
    layer_name: module name of medium mlp layer
    window_size: size of a window to intervene
    window_ids: dict of {layer_name: [window_ids]}
    '''
    layers_to_intervene = window_ids.keys()
    if layer in layers_to_intervene:
        for window_id in window_ids[layer]:
            original_state[:,:,window_id*window_size:(window_id+1)*window_size] = -1*original_state[:,:,window_id*window_size:(window_id+1)*window_size]
    return original_state

def intervene_fn_add(original_state, layer, model_name, window_size, window_ids, steer_vector_path=None, alpha=1):
    '''
    layer_name: module name of medium mlp layer
    window_size: size of a window to intervene
    window_ids: dict of {layer_name: [window_ids]}
    steer_vector_path: path to steer vector tensor ([layers, 4*hidden_size])
    alpha: scalar, strength of intervention
    '''
    layers_to_intervene = window_ids.keys()
    if layer not in layers_to_intervene:
        return original_state

    import numpy as np
    steer_vector_tensor = np.load(steer_vector_path)
    layers = steer_vector_tensor.shape[0]
    assert layers > 0
    steer_vector_dict = {}
    for i in range(layers):
        steer_vector_dict[layer_name_temp[model_name].format(i)] = steer_vector_tensor[i]

    if layer in layers_to_intervene:
        x = original_state
        if isinstance(original_state, tuple):
            original_state = x[0]
        for window_id in window_ids[layer]:
            window_steer_vector = torch.tensor(steer_vector_dict[layer][window_id*window_size:(window_id+1)*window_size], device=original_state.device)
            # print(layer, window_id, original_state.shape)
            # print(steer_vector_dict[layer].shape)
            # print(window_steer_vector.shape)
            original_state[:,:,window_id*window_size:(window_id+1)*window_size] += alpha*window_steer_vector
            del window_steer_vector
    if isinstance(x, tuple):
        return (original_state, x[1])

    return original_state


def get_interveted_output(model, inputs, model_name,
                          window_size, window_ids, alpha=1,
                          intervene_method='zero', steer_vector_path=''):
    '''
    model, inputs, model_name,
    window_size, window_ids, alpha=1,
    intervene_method='zero', steer_vector_path=''
    '''
    intervene_fn = id
    if intervene_method == 'suppress':
        intervene_fn = partial(intervene_fn_zero, window_size=window_size, window_ids=window_ids)
    elif intervene_method == 'neg':
        intervene_fn = partial(intervene_fn_neg, window_size=window_size, window_ids=window_ids)
    elif intervene_method == 'steer':
        intervene_fn = partial(intervene_fn_add, model_name=model_name, window_size=window_size, window_ids=window_ids, 
                               steer_vector_path=steer_vector_path, alpha=alpha)
    layers_to_intervene = window_ids.keys()
    with TraceDict(model, layers_to_intervene, edit_output=intervene_fn) as ret: 
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        elif isinstance(inputs, torch.Tensor):
            outputs = model(inputs)
    return outputs

def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data

def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]

def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn

def get_activations_bau(model, prompt, device=0, model_name='llama7b'):

    model.eval()
    # only focus on the medium activations of mlp
    if model_name == 'llama7b':
        # HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
        # MLPS = [f"model.layers.{i}.mlp.up_proj" for i in range(model.config.num_hidden_layers)]
        LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
        
    elif 'gpt2' in model_name:
        # HEADS = [f"transformer.h.{i}.attn" for i in range(model.config.n_layer)]
        # MLPS = [f"transformer.h.{i}.mlp.c_fc" for i in range(model.config.n_layer)]
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.n_layer)]
    elif 'gpt-neo' in model_name:
        # HEADS = [f"transformer.h.{i}.attn" for i in range(model.config.num_layers)]
        # MLPS = [f"transformer.h.{i}.mlp.c_fc" for i in range(model.config.num_layers)]
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.num_layers)]

    targets = LAYERS
    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, targets) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu()
        # head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        # head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        # mlp_wise_hidden_states = [ret[t].output.squeeze().detach().cpu() for t in MLPS]
        # mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze()

        layer_wise_hidden_states = [ret[t].output[0][0,:].detach().cpu() for t in targets]     # [0] is hidden_states, [bsz(1),seq,hidden_size]
        layer_wise_hidden_states = torch.stack(layer_wise_hidden_states, dim = 0)

    return hidden_states, layer_wise_hidden_states

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = '/data1/models/gpt2'
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # /data/cordercorder/pretrained_models/Qwen/Qwen1.5-0.5B

    import numpy as np
    steer_vector_path = '/home/model_edit/rome/privacy/steering_vector/gpt2-small.age.mean_last.npy'
    window_ids = {'transformer.h.7.mlp.c_fc': [0,2], 'transformer.h.4.mlp.c_fc': [1,2], 'transformer.h.1.mlp.c_fc': [0,3]}

    # input_ids = tokenizer.encode('hello world', return_tensors='pt')
    inputs = tokenizer('hello world', return_tensors="pt")

    output_intervened = get_interveted_output(model, inputs=inputs, model_name='gpt2-small',
                                              window_size=768, window_ids=window_ids,
                                              intervene_method='steer', steer_vector_path=steer_vector_path)
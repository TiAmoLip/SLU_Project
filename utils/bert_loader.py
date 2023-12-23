import json
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
from transformers import BertTokenizer,BertModel
from copy import deepcopy

def encoder(tokenizer,add_vob_path,text_list,max_len = 100):
    #vob_list = tokenizer.get_vocab()
    #with open(add_vob_path, 'r', encoding='utf-8') as f:
    #    lines = f.readlines()
    #new_tokens = []
    #for line in lines:
    #    vob = line.split(' ')[0]
    #    if vob not in vob_list:
    #        new_tokens.append(vob)
    #tokenizer.add_tokens(new_tokens = new_tokens)
    token = tokenizer(
        text_list,
        padding = True,
        truncation = True,
        max_length = max_len,
        return_tensors='pt',
        return_length = True
        )
    input_ids = token['input_ids']
    token_type_ids = token['token_type_ids']
    attention_mask = token['attention_mask']
    leng = token['length']
    length = len(input_ids[0])
    return input_ids,token_type_ids,attention_mask,length,leng

def padding(list, length):
    new_list = deepcopy(list)
    new_list.extend([0]*(length-len(list)))
    return new_list

def bert_load(ex_list, args, tokenizer):
    utt = [ex.utt for ex in ex_list]
    input_ids,token_type_ids,attention_mask,length,leng = encoder(tokenizer, args.word2vec_path, utt)
    args.length = length
    label = [padding(ex.tag_id, length) for ex in ex_list]
    label = torch.tensor(label)
    return TensorDataset(input_ids,token_type_ids,attention_mask,label,leng)
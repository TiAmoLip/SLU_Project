# coding=utf8
from transformers import BertModel
import os
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class BertBasedModel(nn.Module):
    def __init__(self,config,path="./checkpoints/bert",tag_classes=74) -> None:
        super().__init__()
        self.num_tags = tag_classes
        self.config = config
        if os.path.exists(path):
            self.bert = BertModel.from_pretrained(path)
        else:
            self.bert = BertModel.from_pretrained("bert-base-chinese")
            self.bert.save_pretrained(path)
        self.bert.requires_grad_(False)
        self.att = nn.MultiheadAttention(768,8)
        self.rnn = nn.GRU(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.intent = IntentAugment(tag_classes = tag_classes)
        self.loss = nn.CrossEntropyLoss(ignore_index = config.tag_pad_idx)

    def forward(self,batch):
        # x: (bs, seq_len), 是1,2,3,4这种映射过了的
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        x = self.bert(input_ids=input_ids,attention_mask = tag_mask)
        
        # x = self.bert(input_ids=x)
        last_hidden = x['last_hidden_state'] # 这里得到的就是每个词的表示和整个语义信息的高维表示(bs,seq_len,768), (bs, 768)
        # h_prime = self.att(last_hidden,last_hidden,last_hidden)[0]
        packed_inputs = rnn_utils.pack_padded_sequence(last_hidden, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        n_hidden, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        last_hidden = self.att(last_hidden,n_hidden,n_hidden)[0]
        logits = self.intent(last_hidden,None)# bs, seq_len,tag_classes, 这里的None本来应该是h_prime
        
        logits += (1-tag_mask).unsqueeze(-1).repeat(1,1,self.num_tags)* -1e32
        prob = torch.softmax(logits,dim = -1)
        if tag_ids is not None:
            loss = self.loss(logits.view(-1,logits.shape[-1]),tag_ids.view(-1))
            return prob,loss
        return (prob,)
        
        
    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()

        
class IntentAugment(nn.Module):
    def __init__(self, tag_classes) -> None:
        super().__init__()
        # self.att = nn.MultiheadAttention(768*2,8)
        self.mlp = nn.Sequential(
            # nn.Linear(768*3,768),
            nn.Linear(768,768),
            nn.LeakyReLU(0.2),
            nn.Linear(768,tag_classes)
        )
    def forward(self,bert_hidden, att_hidden):
        # h_prime_prime = torch.cat([bert_hidden,att_hidden],dim = -1)
        # e_s = self.att(h_prime_prime,h_prime_prime,h_prime_prime)[0]
        # h_prime_prime = torch.cat([e_s,att_hidden],dim = -1)
        return self.mlp(bert_hidden)

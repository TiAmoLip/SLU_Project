#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)


    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        # pack padded sequence 是因为你输入了一堆0，然后扔给rnn导致烂掉,所以输入的时候要进行压缩，输出的时候要pad回去
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]# bs,seq_len?,num_tags: 32,20,74
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


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )


        
    
class Adapter(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.Wf = nn.Parameter(torch.randn())
        # 这里可以考虑dilated conv或者多层次conv或者拿个矩阵，但我现在还没想好
        


class CharWordFusion(nn.Module):
    def __init__(self, char_vocab_size,word_vocab_size,embed_size,hidden_size,num_tags,pad_id) -> None:
        super().__init__()
        # self.char_len = char_len
        # self.word_len = word_len
        self.embed_size = embed_size
        
        self.char_level = nn.ModuleDict({
            "char_embed": nn.Embedding(char_vocab_size,embed_size),
            "char_lstm": nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1,bidirectional=True),
            "char_project":nn.Linear(hidden_size*2,hidden_size),
            "char_attention": nn.MultiheadAttention(embed_dim=embed_size,num_heads=4,dropout=0.1,batch_first=True)
        })
        
        self.word_level = nn.ModuleDict({
            "word_embed": nn.Embedding(word_vocab_size,embed_size),
            "word_lstm": nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1,bidirectional=True),
            "word_project":nn.Linear(hidden_size*2,hidden_size),
            "word_attention": nn.MultiheadAttention(embed_dim=embed_size,num_heads=4,dropout=0.1,batch_first=True),
            # "word_conv": nn.Conv1d(word_len,char_len,3,1,1)
        })
        self.fuse = nn.MultiheadAttention(embed_dim=embed_size,num_heads=4,dropout=0.1,batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_size+2*hidden_size, hidden_size),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(hidden_size, num_tags),
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    def forward(self,batch):
        
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        char_ids = batch.input_ids
        lengths = batch.lengths
        word_ids= batch.word_ids
        
        
        char_emb = self.char_level['char_embed'](char_ids)
        char_hidden ,_ = self.char_level['char_lstm'](char_emb)
        char_hidden = self.char_level['char_project'](char_hidden)
        char_hidden = torch.cat(self.char_level['char_attention'](char_emb,char_emb,char_emb)[0],char_hidden,dim=-1)
        
        word_emb = self.word_level['word_embed'](word_ids)
        word_hidden,_ = self.word_level['word_lstm'](word_emb)
        word_hidden = self.word_level['word_project'](word_hidden)
        word_hidden = torch.cat(self.word_level['word_attention'](word_emb,char_emb,char_emb)[0],word_hidden,dim=-1)
        
        # char hidden: (bs, char_seq, embed_size), word hidden: (bs, word_seq, embed_size)
        # how to merge charseq and wordseq?
        # 1. concat
        hidden = torch.cat((char_hidden,word_hidden),dim=1)
        hidden = self.fuse(hidden,hidden,hidden)[0][:,:self.word_len,:]
        logits = self.output_layer(hidden)
        logits += (1 - tag_mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if tag_ids is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), tag_ids.view(-1))
            return prob, loss
        return (prob, )
    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]# bs,seq_len?,num_tags: 32,20,74
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
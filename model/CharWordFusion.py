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


class OutputLayer(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(OutputLayer, self).__init__()
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
        
class rnn_attn(nn.Module):
    def __init__(self, vocab_size,embed_size,hidden_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1,bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=embed_size,num_heads=4,dropout=0.1,batch_first=True)
        self.project = nn.Sequential(
            nn.Linear(2*hidden_size,hidden_size),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(hidden_size,hidden_size)
        )
        self.project2 = nn.Linear(embed_size,hidden_size)
    
    def forward(self,input_ids,input_lengths):
        char_emb = self.embed(input_ids)
        
        char_hidden1 = CharWordFusion.pack_and_unpack(char_emb.clone(),input_lengths,self.lstm)
        char_hidden1 = self.project(char_hidden1)
        print(f"char_emb: {char_emb.shape}")
        attn_mask = CharWordFusion.length_to_mask(char_emb.shape[1],input_lengths)
        print("attn_mask: {}".format(attn_mask.shape))
        char_hidden2 = self.attn(char_emb,char_emb,char_emb,attn_mask=attn_mask)[0]

        return torch.cat([char_hidden1,char_hidden2],dim=-1)

class CharWordFusion(nn.Module):
    def __init__(self, char_vocab_size,word_vocab_size,embed_size,hidden_size,num_tags,pad_id) -> None:
        super().__init__()
        # self.char_len = char_len
        # self.word_len = word_len
        self.embed_size = embed_size
        self.num_tags = num_tags
        self.char_level = rnn_attn(char_vocab_size,embed_size,hidden_size)
        self.word_level = rnn_attn(word_vocab_size,embed_size,hidden_size)

        self.fuse = nn.MultiheadAttention(embed_dim=embed_size+hidden_size,num_heads=4,dropout=0.1,batch_first=True)

        self.output = OutputLayer(embed_size+hidden_size,num_tags,pad_id)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    @staticmethod
    def pack_and_unpack(input_embedding, input_lengths,rnn):
        l = sorted(input_lengths,reverse=True)
        packed_inputs = rnn_utils.pack_padded_sequence(input_embedding.clone(),torch.tensor(l).to(torch.device("cpu")),batch_first=True,enforce_sorted=True)
        packed_rnn_out, _ = rnn(packed_inputs)
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out,batch_first=True)
        return rnn_out

    @staticmethod
    def length_to_mask(max_len, lengths):
        mask = torch.tensor([[1]*lengths[i] + [0]*(max_len - lengths[i]) for i in range(max_len)],dtype=torch.bool)
        return mask
    def forward(self,batch):
        # torch.save(batch,"example.pt")
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        char_ids = batch.input_ids
        char_lengths = batch.char_lengths
        word_ids= batch.word_ids
        word_lengths = batch.word_lengths
        
        assert len(char_ids)==len(char_lengths)
        
        char_hidden = self.char_level(char_ids,char_lengths)
        word_hidden = self.word_level(word_ids,word_lengths)
    
    
        hidden = torch.cat([char_hidden,word_hidden],dim=1)

        fuse_mask = self.length_to_mask(hidden.shape[1], list(char_lengths[:char_hidden.shape[1]])+list(word_lengths[:word_hidden.shape[1]]))

        hidden = self.fuse(hidden,hidden,hidden)[0][:,:tag_ids.shape[1],:]

        return self.output(hidden, tag_mask, tag_ids)
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
        
        

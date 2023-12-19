import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
from transformers import BertTokenizer,BertModel

class Bert_base_model(nn.Module):

    def __init__(self,args):
        super(Bert_base_model,self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        if args.frozon:
            for param in self.bert.parameters():
                param.requires_grad_(False)
        self.out = args.num_tags
        self.length = args.length
        self.f1 = nn.Linear(768, self.out)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids,token_type_ids,attention_mask):
        bert_out = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        y = bert_out[0]
        y = self.dropout_layer(y)
        y = self.relu(y)
        y = self.f1(bert_out[0])
        return y.view(input_ids.shape[0],-1,self.out)
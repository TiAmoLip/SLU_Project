from torch.optim import Adam
import torch
import numpy as np
from torch import nn

def evaluate(predictions,labels,evaluator):
    metrics = evaluator.acc(predictions,labels)
    return metrics

def decode(length, tag_ids, label_vocab, utt):
    predictions = []
    for i in range(tag_ids.shape[0]):
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = tag_ids[i][:length[i]]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([utt[i][j] for j in idx_buff])
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
                value = ''.join([utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
    return predictions

def train(args, model, tokenizer, dataloader, loss_fct, optimizer, label_vocab, evaluator):
    model.train()
    losses = []
    accs = []
    utts = []
    label_tags = []
    preds = []
    device = args.device
    for input_ids,token_type_ids,attention_mask,label,length in dataloader:
        utt = []
        for i in range(input_ids.shape[0]):
            utt.append(tokenizer.decode(input_ids[i])[:length[i]])
        utts.append(utt)
        label_tags.extend(decode(length, label.numpy(), label_vocab, utt))
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        out_put = model(input_ids,token_type_ids,attention_mask)
        labels = nn.functional.one_hot(label, num_classes = args.num_tags).to(args.device).type(torch.float32)
        loss = loss_fct(out_put,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = np.argmax(out_put.cpu().detach(),axis=2)
        preds.extend(decode(length, pred.numpy(), label_vocab, utt))
        losses.append(loss.detach().cpu().numpy())
        acc = (np.sum(pred.numpy()==label.numpy())-np.sum((label.numpy()==0)*(pred.numpy()==label.numpy())))/(pred.shape[1]*pred.shape[0]-np.sum(label.numpy()==0))
        accs.append(acc)
    metrics = evaluate(preds, label_tags, evaluator)
    return pred, np.mean(np.array(losses)), np.mean(np.array(accs)), metrics
 
def dev(args, model, tokenizer, dataloader, loss_fct, label_vocab, evaluator):
    model.eval()
    losses = []
    accs = []
    utts = []
    label_tags = []
    preds = []
    device = args.device
    for input_ids,token_type_ids,attention_mask,label,length in dataloader:
        utt = []
        for i in range(input_ids.shape[0]):
            utt.append(tokenizer.decode(input_ids[i])[:length[i]])
        utts.append(utt)
        label_tags.extend(decode(length, label.numpy(), label_vocab, utt))
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        out_put = model(input_ids,token_type_ids,attention_mask)
        labels = nn.functional.one_hot(label, num_classes = args.num_tags).to(args.device).type(torch.float32)
        loss = loss_fct(out_put,labels)
        pred = np.argmax(out_put.cpu().detach(),axis=2)
        preds.extend(decode(length, pred.numpy(), label_vocab, utt))
        losses.append(loss.detach().cpu().numpy())
        acc = (np.sum(pred.numpy()==label.numpy())-np.sum((label.numpy()==0)*(pred.numpy()==label.numpy())))/(pred.shape[1]*pred.shape[0]-np.sum(label.numpy()==0))
        accs.append(acc)
    metrics = evaluate(preds, label_tags, evaluator)
    return pred, np.mean(np.array(losses)), np.mean(np.array(accs)), metrics


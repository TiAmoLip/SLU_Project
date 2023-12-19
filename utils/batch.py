#-*- coding:utf-8 -*-
import torch


def from_example_list(args, ex_list, device='cpu', train=True):
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    # ex_list: list[Example]
    # ex_list[0] : .utt: '导航到贵州省炉盘水市水城县双水卢高速路口'; .tags: ['B-inform-操作', 'I-inform-操作', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...]    
        
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list] # input idx极有可能是将文字转换为对应的id
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens
    batch.did = [ex.did for ex in ex_list]

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]# slot value: ['inform-操作-导航', 'inform-终点名称-贵州省炉盘水市水城县双水六六高速路口']
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list] # tags: ['B-inform-操作', 'I-inform-操作', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...]

        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]# mask: 若被填充则0，否则为1
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        tag_mask = [[1] * len(ex.input_idx) + [0] * (max_len - len(ex.input_idx)) for ex in ex_list]
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
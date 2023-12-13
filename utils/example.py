import json
import pdb

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
# “B-X”表示此元素所在的片段属于X类型并且此元素在此片段的开头。

# “I-X”表示此元素所在的片段属于X类型并且此元素在此片段的中间位置。

# “O”表示不属于任何类型。
class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):# utt: {'utt_id': 1, 'manual_transcript': '我现在在哪里', 'asr_1best': '我现在在哪里', 'semantic': []}
                ex = cls(utt, f'{di}-{ui}')
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']# 这个是整个句子
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'# 
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            # 神奇，self.slot是个字典，而slot竟然是键
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'# 这个B-slot难道是开始的意思?
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

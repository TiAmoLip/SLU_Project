import sys, os, time, gc, json
from torch.optim import Adam
from torch.utils.data import TensorDataset,DataLoader
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from tqdm import tqdm
from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD,LabelVocab
from model.slu_baseline_tagging import SLUTagging
from model.bert_base_model import Bert_base_model
from transformers import BertTokenizer
from utils.bert_loader import *
from scripts.train import train,dev
# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path if args.load_embedding else None)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
label_vocab = Example.label_vocab
evaluator = Example.evaluator
tokenizer = BertTokenizer(args.bert_vob)
train_loader = DataLoader(dataset=bert_load(train_dataset,args,tokenizer), batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(bert_load(dev_dataset, args, tokenizer), batch_size=args.batch_size, shuffle=True)

print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size#1741
args.pad_idx = Example.word_vocab[PAD]#0
args.num_tags = Example.label_vocab.num_tags#74
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)#0

model = Bert_base_model(args)
model.to(device)
loss_fct = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr = args.lr)
train_losses = []
train_accs = []
dev_losses = []
dev_accs = []
best_result = {}
best_result['dev_acc'] = 0
with tqdm(total = args.max_epoch) as pbar:
    for i in range(args.max_epoch):
        pred, train_loss, train_acc, train_metrics = train(args,model,tokenizer,train_loader,loss_fct,optimizer,label_vocab,evaluator)
        train_accs.append(train_acc),train_losses.append(train_loss)
        pred, dev_loss, dev_acc, dev_metrics = dev(args,model,tokenizer,dev_loader,loss_fct,label_vocab,evaluator)
        dev_accs.append(dev_acc),dev_losses.append(dev_loss)
        pbar.set_description('t_l: %.2f t_a1: %.2f t_a2: %.2f t_s(p/r/f): (%.2f/%.2f/%.2f) d_l: %.2f d_a1: %.2f d_a2: %.2f d_s(p/r/f):(%.2f/%.2f/%.2f)'%\
                            (train_loss,train_acc,train_metrics['acc'],\
                            train_metrics['fscore']['precision'],train_metrics['fscore']['recall'],\
                            train_metrics['fscore']['fscore'],dev_loss,dev_acc,dev_metrics['acc'],\
                            dev_metrics['fscore']['precision'],dev_metrics['fscore']['recall'],dev_metrics['fscore']['fscore']))
        if dev_metrics['acc'] > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_metrics['acc'], dev_metrics['fscore'], i
        pbar.update(1)
    print("Evaluation costs %.2fs, Best result epoch: %d ; Dev loss: %.2f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, \
                                                            best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], \
                                                            best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
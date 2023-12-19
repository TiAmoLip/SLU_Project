This is the final project of CS3602 Natural Language Processing in SJTU, whose subject is Spoken Language Processing

### baseline:
```bash
FINAL BEST RESULT:  Epoch: 4  Dev loss: 0.6676  Dev acc: 71.8436  Dev fscore(p/r/f): (79.8883/74.5568/77.1305)
```
by running:
```bash
python scripts/slu_baseline.py --load_embedding True
```

### Bert:
```bash
Evaluation costs 1590.08s, Best result epoch: 32 ; Dev loss: 1.39        Dev acc: 81.23  Dev fscore(p/r/f): (82.35/90.21/86.10)
```
by running:
```bash
python scripts/bert_base.py --max_epoch 50
```
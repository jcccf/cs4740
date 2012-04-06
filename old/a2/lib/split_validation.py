import random, sys

random.seed(1234)

if len(sys.argv) > 1:
    split = float(sys.argv[1])
else:
    split = 0.7

with open('data/wsd-data/train.data','r') as f:
    l = f.readlines()
    random.shuffle(l)
split = int( split*len(l) )
train = l[0:split]
validation = l[split:]
with open('data/wsd-data/train_split.data','w') as f:
    for line in train:
        f.write(line)
with open('data/wsd-data/valiation_split.data','w') as f:
    for line in validation:
        f.write(line)

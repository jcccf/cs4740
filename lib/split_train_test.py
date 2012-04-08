import Parser, HMM, random
random.seed(1023)

data = Parser.parse_training_file()
test_data = Parser.parse_test_file()
train_len, test_len = len(data), len(test_data)
print train_len, test_len

split = int((train_len - 0.0) * 0.8)
random.shuffle(data)
train_data, val_data = data[:split], data[split:]

with open('data/pos_files/development.pos','w') as f:
    for sequence in train_data:
        for pos,word in sequence:
            f.write(pos+" "+word+"\n")
with open('data/pos_files/validation.pos','w') as f:
    for sequence in val_data:
        for pos,word in sequence:
            f.write(pos+" "+word+"\n")

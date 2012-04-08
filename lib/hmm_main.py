# HMM Main
import Parser, HMM, random
from sys import stdout
random.seed(1023)

data = Parser.parse_training_file()
test_data = Parser.parse_test_file()
train_len, test_len = len(data), len(test_data)
print train_len, test_len

split = int((train_len - 0.0) * 0.8)
random.shuffle(data)
train_data, val_data = data[:split], data[split:]

for ngram in [2]:
  for smooth in ["none"]:
    print "NGram: ", ngram, " Smooth: ", smooth
    
    hmm = HMM.HMM(ngram=ngram, smooth=smooth)
    hmm.train(train_data)
    print "Done Training"
    correct, total = 0, 0
    for eg in val_data:
      eg_pos, eg_words = zip(*eg)
      seq, prob = hmm.decode_fast(eg_words)
      correct += sum([1 for a,b in zip(seq, eg_pos) if a == b])
      total += len(eg_pos)
      stdout.write(".")
      stdout.flush
    
    print
    print "Correct:",correct
    print "Total:",total
    print "Accuracy:",float(correct)/total
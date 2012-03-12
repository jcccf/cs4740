import Parser

class Baselinemostfrequentsense:
    def __init__(self):
        self.sensecounts = dict()
        
    def create_sensecounts(self, egs):
        for example in egs:
            if (example.word) in self.sensecounts:
                self.sensecounts[(example.word)] = [sum(sensespair) for sensespair in zip(self.sensecounts[example.word], example.senses)]
            else:
                self.sensecounts[(example.word)] = example.senses
                
    def predict_sense(self, testexample):
        if (testexample.word) not in self.sensecounts:
            print(testexample.word + " not in training set")
            # just choose the first sense since no way to predict without any data for the word
            return [(1 if i == 0 else 0) for i in range(len(testexample.senses))]
        else:
            counts = self.sensecounts[(testexample.word)]
            predictedindex = counts.index(max(counts))
            # print(testexample.word + " " + str(predictedindex))
            if predictedindex >= len(testexample.senses):
                print(testexample.word + "test set has less senses than training set")
                return [(1 if i == 0 else 0) for i in range(len(testexample.senses))]
            else:
                return [(1 if i == predictedindex else 0) for i in range(len(testexample.senses))]

if __name__ == '__main__':
    classifier = Baselinemostfrequentsense()
    egs = Parser.load_examples('data/wsd-data/train_split.data')
    classifier.create_sensecounts(egs)
    testdata = Parser.load_examples('data/wsd-data/valiation_split.data')
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for testexample in testdata:
        prediction = classifier.predict_sense(testexample)
        for (k,(s,p)) in enumerate(zip(testexample.senses, prediction)):
            if s == 1 and p == 1:
                tp += 1
            elif s == 0 and p == 0:
                tn += 1
            elif s == 1 and p == 0:
                fn += 1
            elif s == 0 and p == 1:
                fp += 1
    prec = tp/(tp+fp)
    rec  = tp/(tp+fn)
    f1   = 2.0*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
    print "%d, %d, %d, %d\n%f, %f, %f"%(tp,tn,fp,fn,prec,rec,f1)
print("done")
        
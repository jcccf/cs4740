import Parser

class Baselinemostfrequentsense:
    def __init__(self):
        self.sensecounts = dict()
        
    def create_sensecounts(self, egs):
        for example in egs:
            if (example.word, example.pos) in self.sensecounts:
                self.sensecounts[(example.word, example.pos)] = [sum(sensespair) for sensespair in zip(self.sensecounts[(example.word, example.pos)], example.senses)]
            else:
                self.sensecounts[(example.word, example.pos)] = example.senses
                
    def predict_sense(self, testexample):
        if (testexample.word, testexample.pos) not in self.sensecounts:
            print(testexample.word + "not in test set")
            # just choose the first sense since no way to predict without any data for the word
            return [(1 if i == 0 else 0) for i in range(len(testexample.senses))]
        else:
            counts = self.sensecounts[(testexample.word, testexample.pos)]
            predictedindex = counts.index(max(counts))
            # print(testexample.word + " " + str(predictedindex))
            if predictedindex >= len(testexample.senses):
                print(testexample.word + "test set has less senses than training set")
                return [(1 if i == 0 else 0) for i in range(len(testexample.senses))]
            else:
                return [(1 if i == predictedindex else 0) for i in range(len(testexample.senses))]

if __name__ == '__main__':
    classifier = Baselinemostfrequentsense()
    egs = Parser.load_examples()
    classifier.create_sensecounts(egs)
    prediction = classifier.predict_sense(egs[0])
    print(prediction)
        
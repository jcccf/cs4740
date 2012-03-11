import Parser
import nltk
         
def features(example):
    return {'word_before': example.context_before.split()[-1], 
            'word_after': example.context_after.split()[0] }

def disambiguate():
    dictegs = Parser.load_training_data('data/wsd-data/train.data')
    testdata = Parser.load_test_data('data/wsd-data/test.data')
    currentmodel = ('','')
    with open('data/output/classifier.txt', 'w') as f:
        for testexample in testdata:
            if currentmodel != (testexample.word, testexample.pos):
                # train classifier for each sense of word and pos
                examples = dictegs[(testexample.word, testexample.pos)]
                classifiers = []
                for senseno in range(len(testexample.senses)):
                    train_feature_sets = [(features(ex), ex.senses[senseno]) for ex in examples]
                    # temporarily used Naive Bayes
                    # replace with whatever pro classifier that you like
                    classifiers.append(nltk.NaiveBayesClassifier.train(train_feature_sets))
                currentmodel = (testexample.word, testexample.pos)
            # classifier has been trained for word and pos                    
            for senseno in range(len(testexample.senses)):
                prediction = classifiers[senseno].classify(features(testexample))
                f.write('%d\n' % prediction)
        f.close()
    print("done")

if __name__ == "__main__":
    disambiguate()
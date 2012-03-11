import Parser, Baselinemostfrequentsense, os
import weka_classifier
import nltk.classify

print "==CS 4740 Project 2=="
task = int(input("select method: (baseline most frequent sense=1) (naive_bayes=2) (max_ent=3) "))

if task == 1:
    try:
        os.makedirs('data/output/')
    except:
        pass
    classifier = Baselinemostfrequentsense.Baselinemostfrequentsense()
    egs = Parser.load_examples()
    classifier.create_sensecounts(egs)
    testdata = Parser.load_data('data/wsd-data/test.data')
    with open('data/output/mostfreqsense.txt', 'w') as f:
        for testexample in testdata:
            prediction = classifier.predict_sense(testexample)
            for element in prediction:
                f.write('%d\n' % element)
        f.close()
    print("done")
elif task == 2 or task == 3:
    try:
        os.makedirs('data/output/')
    except:
        pass
    if task == 2:
        classifier = weka_classifier.weka_classifier(window_size=10, classifier=nltk.NaiveBayesClassifier)
    elif task == 3:
        classifier = weka_classifier.weka_classifier(window_size=10, classifier=nltk.ConditionalExponentialClassifier)

    egs = Parser.load_examples()
    classifier.train(egs)
    testdata = Parser.load_data('data/wsd-data/test.data')
    with open('data/output/mostfreqsense.txt', 'w') as f:
        for testexample in testdata:
            prediction = classifier.predict([testexample])
            for element in prediction:
                f.write('%d\n' % element)
        f.close()
    print("done")

        
    

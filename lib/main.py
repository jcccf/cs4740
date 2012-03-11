
import Parser, Baselinemostfrequentsense, Disambiguation, os
import weka_classifier
import nltk.classify

print "==CS 4740 Project 2=="
task = int(input("select method: (baseline most frequent sense=1, classifier=2, naive_bayes=3, max_ent=4) "))

if task == 1:
    try:
        os.makedirs('data/output/')
    except:
        pass
    classifier = Baselinemostfrequentsense.Baselinemostfrequentsense()
    egs = Parser.load_examples()
    classifier.create_sensecounts(egs)
    testdata = Parser.load_test_data('data/wsd-data/test.data')
    with open('data/output/mostfreqsense.txt', 'w') as f:
        for testexample in testdata:
            prediction = classifier.predict_sense(testexample)
            for element in prediction:
                f.write('%d\n' % element)
        f.close()
    print("done")
elif task == 2:
    Disambiguation.disambiguate()    
elif task == 3 or task == 4:
    try:
        os.makedirs('data/output/')
    except:
        pass
    if task == 3:
        classifier = weka_classifier.weka_classifier(
            window_size=10,
            classifier=nltk.NaiveBayesClassifier,
            split_pre_post=False)
        name = "NaiveBayes"
    elif task == 4:
        classifier = weka_classifier.weka_classifier(
            window_size=10,
            classifier=nltk.ConditionalExponentialClassifier,
            split_pre_post=False)
        name = "MaxEnt"

    egs = Parser.load_examples()
    classifier.train(egs)
    #testdata = Parser.load_data('data/wsd-data/test.data')
    testdata = Parser.load_test_data('data/wsd-data/test.data')
    with open('data/output/%s.txt'%name, 'w') as f:
        for testexample in testdata:
            prediction = classifier.predict([testexample])
            for element in prediction:
                f.write('%d\n' % element)
        f.close()
    print("done")

        
    

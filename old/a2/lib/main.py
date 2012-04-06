
import Parser, Baselinemostfrequentsense, Disambiguation, os
import weka_classifier
import nltk.classify
import scikit_classifier

print "==CS 4740 Project 2=="
task = int(input("select method: (baseline most frequent sense=1, naive_bayes=3, max_ent=4, scikit=5) "))

try:
    os.makedirs('data/output/')
except:
    pass
    
if task == 1:
    classifier = Baselinemostfrequentsense.Baselinemostfrequentsense()
    egs = Parser.load_examples()
    classifier.create_sensecounts(egs)
    testdata = Parser.load_examples('data/wsd-data/test.data')
    with open('data/output/mostfreqsense.txt', 'w') as f:
        for testexample in testdata:
            prediction = classifier.predict_sense(testexample)
            for element in prediction:
                f.write('%d\n' % element)
        f.close()
    print("done")  
elif task == 3 or task == 4:
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
elif task == 5:
    name = "scikit_test_ws500_pos1_ngram0_synfeat0_uselesk_useleskwords"
    classifier = scikit_classifier.scikit_classifier(
        window_size=500,
        use_syntactic_features=0,
        pos_window_size=1,
        ngram_size=0,
        use_lesk=True,
        use_lesk_words=True,
        training_file='data/wsd-data/train.data',
        test_file='data/wsd-data/test.data')
    egs = Parser.load_examples('data/wsd-data/train.data')
    test_egs = Parser.load_examples('data/wsd-data/test.data')
    
    # Train the classifier(s)
    classifier.train(egs)
    with open('data/output/%s.txt'%name, 'w') as f:
        for eg in test_egs:
            pred = classifier.predict([eg])
            for p in pred:
                f.write("%d\n"%p)



















import Parser, Baselinemostfrequentsense, Disambiguation, os

print "==CS 4740 Project 2=="
task = int(input("select method: (baseline most frequent sense=1, classifier=2) "))

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
        
    
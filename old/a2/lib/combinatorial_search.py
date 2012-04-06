import Parser, scikit_classifier

if __name__ == '__main__':
    # takes forever to run
    with open('data/output/searchresult.txt', 'w') as f:
        for ngram_size in [0,1,2]:
            for use_syntactic_features in [0]:
                for pos_window_size in [0,1,5,10]:
                    for window_size in [0, 5, 10, 50, 100, 500]:
                        for use_lesk in [False]:
                            for use_lesk_words in [False]:
                                classifier = scikit_classifier.scikit_classifier(
                                                pos_window_size = pos_window_size,
                                                ngram_size = ngram_size,
                                                window_size = window_size,
                                                use_syntactic_features = use_syntactic_features,
                                                use_lesk = use_lesk,
                                                use_lesk_words = use_lesk_words)
                            
                                egs = Parser.load_examples('data/wsd-data/train_split.data')
                                test_egs = Parser.load_examples('data/wsd-data/valiation_split.data')
                            
                                classifier.train(egs)                        
                                tp = 0.0
                                fp = 0.0
                                tn = 0.0
                                fn = 0.0
                                for eg in test_egs:
                                    pred = classifier.predict([eg])
                                    for (k,(s,p)) in enumerate(zip(eg.senses,pred)):
                                        if s == 1 and p == 1:
                                            tp += 1
                                        elif s == 0 and p == 0:
                                            tn += 1
                                        elif s == 1 and p == 0:
                                            fn += 1
                                        elif s == 0 and p == 1:
                                            fp += 1
                                prec = tp/(tp+fp) if tp+fp > 0 else 0
                                rec  = tp/(tp+fn) if tp+fn > 0 else 0
                                f1   = 2.0*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
                                print "%d %d %d %d %d %d %f %f %f\n"%(ngram_size,pos_window_size,window_size,use_syntactic_features,use_lesk,use_lesk_words,prec,rec,f1)
                                f.write("%d %d %d %d %d %d %f %f %f\n"%(ngram_size,pos_window_size,window_size,use_syntactic_features,use_lesk,use_lesk_words,prec,rec,f1))
                            
                            
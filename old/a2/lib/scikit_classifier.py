import Parser, sys, os, string, MVectorizer, Syntactic_features, Baselinemostfrequentsense
from optparse import OptionParser
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import RidgeClassifier
from sklearn.svm.sparse import LinearSVC
# from sklearn.linear_model.sparse import SGDClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from NGramModel import NGramModel

import scipy.sparse as sps # sps.csr_matrix, sps.hstack

class scikit_classifier:
    def __init__(self,pos_window_size=1,ngram_size=0,window_size=3,use_syntactic_features=0,use_lesk=False,lesk_window_size=100,use_lesk_words=False,lesk_words_window_size=2,training_file='data/wsd-data/train_split.data',test_file='data/wsd-data/valiation_split.data'):
        self.vectorizers = {}
        self.pos_vectorizers = {}
        self.lesky_words_vectorizers = {}
        self.syn_vectorizers = {}
        self.classifiers = {}
        self.pos_window_size = pos_window_size
        self.ngram_size = ngram_size
        self.use_syntactic_features = use_syntactic_features
        self.use_lesk = use_lesk
        self.lesk_window_size = lesk_window_size
        self.use_lesk_words = use_lesk_words
        self.lesk_words_window_size = lesk_words_window_size
        self.window_size = window_size
        self.training_file = training_file
        self.test_file = test_file
        
    def prepare_examples(self, egs, for_training=True, verbose=False):
        dictionary = Parser.load_dictionary()
      
        # Prepares the examples into training data, applying features etc.
        if verbose:
            print "Preparing %d examples"%len(egs),
        data, labels, pos, ngram, nsenses, syntactic, lesky, lesky_words = {}, {}, {}, {}, {}, {}, {}, {}
        if (self.use_syntactic_features and for_training):
                word_list = Syntactic_features.prepare_file(self.training_file)
                syn_train = Syntactic_features.parse_stanford_output(self.training_file, word_list)
                syn_index = 0
        for eg in egs:
            if verbose:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            eg.word = eg.word.lower()
            if not eg.word in data:
                data[eg.word] = []
                labels[eg.word] = []
                pos[eg.word] = []
                lesky[eg.word] = []
                lesky_words[eg.word] = []
                if (self.use_syntactic_features and for_training):
                    syntactic[eg.word] = []
            # text = eg.context_before + " " + eg.target + " " + eg.pos + " " + eg.context_after
            #text = eg.context_before + " " + eg.target + " " + eg.context_after
            pre_words = eg.context_before.lower().split()[-self.window_size:]
            post_words = eg.context_after.lower().split()[:self.window_size]
            text = ' '.join(pre_words) + ' ' + eg.target + ' ' + ' '.join(post_words) # TODO worsens our F1!
            data[eg.word].append( text )
            label = [ idx for idx,val in enumerate(eg.senses) if val == 1 ]
            labels[eg.word].append( label )
            pos[eg.word].append(eg.pos_positions(window=self.pos_window_size))
            
            if self.use_lesk:
              lesky[eg.word].append(eg.lesk(dictionary, window_size=self.lesk_window_size))
              
            if self.use_lesk_words:
              lesky_words[eg.word].append(' '.join(eg.lesk_words(dictionary, window_size=self.lesk_words_window_size)))
            
            if (self.use_syntactic_features and for_training):
                syntactic[eg.word].append(syn_train[syn_index])
                syn_index += 1
            if for_training and self.ngram_size > 0:
                if eg.word not in nsenses:
                    nsenses[eg.word] = len(eg.senses)
                    for idx in range(0,len(eg.senses)):
                        ngram[eg.word+str(idx)] = NGramModel(self.ngram_size, smooth_type="lap", unknown_type=None, gram_type="n")
                for idx in label:
                    key = eg.word+str(idx)
                    assert key in ngram
                    # Only laplacian smoothing and no unknowns allows incremental training
                    ngram[key].train([text])
        # print pos
        # raise Exception()
        if for_training:
            return (data, labels, pos, lesky, lesky_words, ngram, nsenses, syntactic)
        else:
            return (data, labels, pos, lesky, lesky_words)
    
    def most_informative_features(self):
        # Returns list of (word, sense_idx, feature_key, feature_value)
        lst = []
        for word,classifiers in self.classifiers.iteritems():
            context_vectorizer = self.vectorizers[word]
            context_len = context_vectorizer.transform([""]).shape[1]
            
            pos_vectorizer = self.pos_vectorizers[word]
            pos_len = pos_vectorizer.transform([""]).shape[1]
            
            # if (self.use_syntactic_features):
                # syn_vectorizer = self.syn_vectorizers[word]
                # syn_len = syn_vectorizer.transform([""]).shape[1]
            
            total_len = context_len + pos_len
            # total_len = context_len
            
            for sense_id,classifier in enumerate(classifiers.estimators_):
                coeff = classifier.coef_.tolist()[0]
                
                ## Context features
                # Find the feature with the highest (abs) weight
                head = coeff[:context_len]
                coeff = coeff[context_len:]
                
                abshead = [abs(x) for x in head]
                feature_value = max(abshead)
                maxid = abshead.index(feature_value)
                feature_value = head[maxid]
                assert maxid < context_len
                # Find the feature key
                feature_key = context_vectorizer.inverse_transform(
                    csr_matrix( ([1], ([0], [maxid])), shape=(1,context_len) ))[0][0]
                context_key,context_val = (feature_key,feature_value)
                
                ## POS features
                # Find the feature with the highest (abs) weight
                head = coeff[:pos_len]
                coeff = coeff[pos_len:]
                
                abshead = [abs(x) for x in head]
                feature_value = max(abshead)
                maxid = abshead.index(feature_value)
                feature_value = head[maxid]
                assert maxid < pos_len
                # Find the feature key
                feature_key = pos_vectorizer.inverse_transform(
                    csr_matrix( ([1], ([0], [maxid])), shape=(1,pos_len) ))[0][0]
                pos_key,pos_val = (feature_key,feature_value)
                
                # If it gets here, it isn't a feature that I know of..
                lst.append( (word,sense_id,context_key,context_val,pos_key,pos_val) )
        return lst

    def train(self,egs):
        # Trains a classifier for each word sense
        data,labels,pos,lesky,lesky_words,ngram,nsenses,syntactic = self.prepare_examples(egs,verbose=True)
        self.ngram = ngram
        self.nsenses = nsenses
        print "\nTraining on %d words"%len(data),
        for word in labels.iterkeys():
            sys.stdout.write(".")
            sys.stdout.flush()
            
            # Extract context features
            self.vectorizers[word] = Vectorizer()
            X = self.vectorizers[word].fit_transform(data[word])
            
            # Add Parts of Speech
            self.pos_vectorizers[word] = Vectorizer()
            X_pos = self.pos_vectorizers[word].fit_transform(pos[word])
            X = sps.hstack((X, X_pos))
              
            # Add Lesky Words
            if self.use_lesk_words:
              self.lesky_words_vectorizers[word] = Vectorizer()
              X_leskwords = self.lesky_words_vectorizers[word].fit_transform(lesky_words[word])
              X = sps.hstack((X, X_leskwords))
            
            # Add Lesky
            if self.use_lesk:
              X_lesk = MVectorizer.rectangularize(lesky[word])
              X = sps.hstack((X, X_lesk))
         
            # Add Syntactic dependencies
            if (self.use_syntactic_features):
                if all(synfeat == [] for synfeat in syntactic[word]):
                    pass
                else:
                    self.syn_vectorizers[word] = MVectorizer.ListsVectorizer()
                    X_syn = self.syn_vectorizers[word].fit_transform(syntactic[word])
                    (x_rows,x_cols) = X.shape
                    (xsyn_rows,xsyn_cols) = X_syn.shape
                    if x_rows != xsyn_rows:
                        X_filler = sps.coo_matrix((x_rows-xsyn_rows,xsyn_cols))
                        X_syn = sps.vstack((X_syn,X_filler))
                    X = sps.hstack((X, X_syn))
            
            # Add NGram model
            if self.ngram_size > 0:
                num_senses = self.nsenses[word]
                ngram_list = []
                for sentence in data[word]:
                    ngram_list.append( dict([ ( idx, self.ngram[word+str(idx)].get_perplexity(sentence,True) ) for idx in range(0,num_senses) ]) )
                X_ngram = MVectorizer.DictsVectorizer().fit_transform(ngram_list)
                X = sps.hstack((X, X_ngram))
                
            Y = labels[word]
            
            # Learn classifier
            # self.classifiers[word] = OneVsRestClassifier(SVC(kernel='linear',scale_C=True)) #Doesn't work
            self.classifiers[word] = OneVsRestClassifier(LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3))
            self.classifiers[word].fit(X,Y)

        print "\nDone"

    def predict(self, egs):
        # Given a list of examples, predict their word senses
        res = []
        if (self.use_syntactic_features):
            word_list = Syntactic_features.prepare_file(self.test_file)
            syntactic = Syntactic_features.parse_stanford_output(self.test_file, word_list)
            syn_index = 0
        for eg in egs:
            eg.word = eg.word.lower()
            data,labels,pos,lesky,lesky_words = self.prepare_examples([eg], for_training=False)
            
            # Add context words
            X = self.vectorizers[eg.word].transform(data[eg.word])
            
            # Add Parts of Speech
            X_pos = self.pos_vectorizers[eg.word].transform(pos[eg.word])
            X = sps.hstack((X, X_pos))
              
            # Add Lesky Words
            if self.use_lesk_words:
              X_leskywords = self.lesky_words_vectorizers[eg.word].transform(lesky_words[eg.word])
              X = sps.hstack((X, X_leskywords))
                
            # Add Lesky
            if self.use_lesk:
              X_lesk = MVectorizer.rectangularize(lesky[eg.word])
              X = sps.hstack((X, X_lesk))
            
            # Add Syntactic dependencies
            if (self.use_syntactic_features):
                if all(synfeat == [] for synfeat in syntactic[syn_index]):
                    pass
                elif (not (eg.word in self.syn_vectorizers)):
                    pass
                else:
                    X_syn = self.syn_vectorizers[eg.word].transform([syntactic[syn_index]])
                    (x_rows,x_cols) = X.shape
                    (xsyn_rows,xsyn_cols) = X_syn.shape
                    if x_rows != xsyn_rows:
                        X_filler = sps.coo_matrix((x_rows-xsyn_rows,xsyn_cols))
                        X_syn = sps.vstack((X_syn,X_filler))
                    X = sps.hstack((X, X_syn))
                syn_index += 1
            
            # Add NGram model
            if self.ngram_size > 0:
                num_senses = self.nsenses[eg.word]
                assert num_senses == len(eg.senses)
                ngram_list = []
                for sentence in data[eg.word]:
                    ngram_list.append( dict([ ( idx, self.ngram[eg.word+str(idx)].get_perplexity(sentence,True) ) for idx in range(0,num_senses) ]) )
                X_ngram = MVectorizer.DictsVectorizer().fit_transform(ngram_list)
                X = sps.hstack((X, X_ngram))
            
            Y = self.classifiers[eg.word].predict(X)
            
            senses = [0]*len(eg.senses)
            for y in list(Y[0]):
                senses[y] = 1
            res.extend(senses)
        return res


if __name__ == '__main__':
    optParser = OptionParser()
    optParser.add_option("--pos_ws", help="Part-of-speech window size",
                  action="store", type="int", dest="pos_window_size", default=1)
    optParser.add_option("--ngram", help="Ngram size",
                  action="store", type="int", dest="ngram_size", default=0)
    optParser.add_option("--ws", help="Context window size",
                  action="store", type="int", dest="window_size", default=500)
    optParser.add_option("--use_syntactic_features", help="Use syntactic features?",
                  action="store", type="int", dest="use_syntactic_features", default=0)
    optParser.add_option("--use_lesk", action="store_true", dest="use_lesk", default=False)
    optParser.add_option("--lesk_ws", help="Lesk window size",
                  action="store", type="int", dest="lesk_window_size", default=100)
    optParser.add_option("--use_lesk_words", action="store_true", dest="use_lesk_words", default=False)
    optParser.add_option("--lw_ws", help="Lesk words window size",
                  action="store", type="int", dest="lesk_words_window_size", default=3)
    optParser.add_option("--most_informative_features", action="store_true", dest="most_informative_features", default=False)
    optParser.add_option("--output", help="Output predicted senses to file",
                  action="store", type="string", dest="outfile", default="")
    optParser.add_option("--merge_with_most_frequent_sense", help="Incorporate most freqent sense?",
                  action="store_true", dest="merge_with_most_frequent_sense", default=False)

    (options,args) = optParser.parse_args()
    print options
    # classifier = weka_classifier(10,nltk.DecisionTreeClassifier)  # Does not work with sparse features
    # classifier = weka_classifier(10,nltk.ConditionalExponentialClassifier,split_pre_post=False)
    # classifier = weka_classifier(10,nltk.NaiveBayesClassifier,split_pre_post=True)

    classifier = scikit_classifier(
                    pos_window_size = options.pos_window_size,
                    ngram_size = options.ngram_size,
                    window_size = options.window_size,
                    use_syntactic_features = options.use_syntactic_features,
                    use_lesk = options.use_lesk,
                    lesk_window_size = options.lesk_window_size,
                    use_lesk_words = options.use_lesk_words,
                    lesk_words_window_size = options.lesk_words_window_size)

    egs = Parser.load_examples('data/wsd-data/train_split.data')
    test_egs = Parser.load_examples('data/wsd-data/valiation_split.data')

    classifier.train(egs)
    
    if options.most_informative_features:
        print "Most informative features (Word, sense_id, context_word, svm_weight, pos_word, svm_weight):"
        for (word,sense_id,context_key,context_val,pos_key,pos_val) in classifier.most_informative_features():
            print word,sense_id,context_key,context_val,pos_key,pos_val
        sys.stdout.flush()
        exit(0)
        
    baseline_classifier = None
    if options.merge_with_most_frequent_sense:
        baseline_classifier = Baselinemostfrequentsense.Baselinemostfrequentsense()
        baseline_classifier.create_sensecounts(egs)

    # prediction = classifier.predict( egs[0:3] )
    # print(prediction)
    # print "True labels vs predicted labels"
    outf = None if options.outfile == "" else open(options.outfile,"w")
    
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for eg in test_egs:
        # print eg.word+" ",
        # print eg.senses,
        # print " vs ",
        # print classifier.predict([eg])
        pred = classifier.predict([eg])
        if baseline_classifier is not None and pred == [0]*len(pred):   # Prediction is empty
            pred = baseline_classifier.predict_sense(eg)
        
        for (k,(s,p)) in enumerate(zip(eg.senses,pred)):
            """ Word \t POS \t Sense # \t True label \t Predicted label """
            if outf is not None:
                outf.write("%s\t%s\t%d\t%d\t%d\n"%(eg.word,eg.pos,k,s,p) )
            #print "%s\t%s\t%d\t%d\t%d"%(eg.word,eg.pos,k,s,p)
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
    print "%d, %d, %d, %d\n%f, %f, %f"%(tp,tn,fp,fn,prec,rec,f1)
    
    if outf is not None:
        outf.close()
        outf = None
    

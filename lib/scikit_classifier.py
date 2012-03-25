import Parser, sys, os, string, MVectorizer

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
    def __init__(self,pos_window_size=1,ngram_size=0):
        self.vectorizers = dict()
        self.pos_vectorizers = dict()
        self.classifiers = dict()
        self.pos_window_size = pos_window_size
        self.ngram_size = ngram_size
        pass
        
    def prepare_examples(self, egs, for_training=True, verbose=False):
        # Prepares the examples into training data, applying features etc.
        if verbose:
            print "Preparing %d examples"%len(egs),
        data, labels, pos, ngram, nsenses = {}, {}, {}, {}, {}
        for eg in egs:
            if verbose:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            eg.word = eg.word.lower()
            if not eg.word in data:
                data[eg.word] = []
                labels[eg.word] = []
                pos[eg.word] = []
            # text = eg.context_before + " " + eg.target + " " + eg.pos + " " + eg.context_after
            text = eg.context_before + " " + eg.target + " " + eg.context_after
            data[eg.word].append( text )
            label = [ idx for idx,val in enumerate(eg.senses) if val == 1 ]
            labels[eg.word].append( label )
            pos[eg.word].append(eg.pos_positions(window=self.pos_window_size))
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
            return (data, labels, pos, ngram, nsenses)
        else:
            return (data, labels, pos)

    def train(self,egs):
        # Trains a classifier for each word sense
        data,labels,pos,ngram,nsenses = self.prepare_examples(egs,verbose=True)
        self.ngram = ngram
        self.nsenses = nsenses
        print "\nTraining on %d words"%len(data),
        for word in labels.iterkeys():
            sys.stdout.write(".")
            sys.stdout.flush()
            # Extract features
            self.vectorizers[word] = Vectorizer()
            X = self.vectorizers[word].fit_transform(data[word])
            
            # Add Parts of Speech
            self.pos_vectorizers[word] = MVectorizer.ListsVectorizer()
            X_pos = self.pos_vectorizers[word].fit_transform(pos[word])
            X = sps.hstack((X, X_pos))
            
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
        for eg in egs:
            eg.word = eg.word.lower()
            data,labels,pos = self.prepare_examples([eg], for_training=False)
            X = self.vectorizers[eg.word].transform(data[eg.word])
            
            # Add Parts of Speech
            X_pos = self.pos_vectorizers[eg.word].transform(pos[eg.word])
            X = sps.hstack((X, X_pos))
            
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
    # classifier = weka_classifier(10,nltk.DecisionTreeClassifier)  # Does not work with sparse features
    # classifier = weka_classifier(10,nltk.ConditionalExponentialClassifier,split_pre_post=False)
    # classifier = weka_classifier(10,nltk.NaiveBayesClassifier,split_pre_post=True)
    classifier = scikit_classifier(pos_window_size=1,ngram_size=0)
    egs = Parser.load_examples('data/wsd-data/train_split.data')
    test_egs = Parser.load_examples('data/wsd-data/valiation_split.data')

    if True:
        classifier.train(egs)
#       prediction = classifier.predict( egs[0:3] )
#       print(prediction)
        # print "True labels vs predicted labels"
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
            # print pred
            for (k,(s,p)) in enumerate(zip(eg.senses,pred)):
                """ Word \t POS \t Sense # \t True label \t Predicted label """
                print "%s\t%s\t%d\t%d\t%d"%(eg.word,eg.pos,k,s,p)
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
    
    if False:
        classifier.prepare_examples(egs,use_str=True)
        try:
            os.makedirs('data/weka/')
        except:
            pass
        classifier.write_arff("data/weka/")

    

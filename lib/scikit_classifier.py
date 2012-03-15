import Parser, sys, os, string

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

import scipy.sparse as sps # sps.csr_matrix, sps.hstack

class scikit_classifier:
    def __init__(self):
        self.vectorizers = dict()
        self.classifiers = dict()
        pass
        
    def prepare_examples(self,egs,verbose=False):
        # Prepares the examples into training data, applying features etc.
        if verbose:
            print "Preparing on %d examples"%len(egs),
        data, labels, pos = {}, {}, {}
        for eg in egs:
            if verbose:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            eg.word = eg.word.lower()
            if not eg.word in data:
                data[eg.word] = []
                labels[eg.word] = []
                pos[eg.word] = []
            data[eg.word].append( eg.context_before + " " + eg.target + " " + eg.context_after )
            labels[eg.word].append( [ idx for idx,val in enumerate(eg.senses) if val == 1 ] )
            pos[eg.word].append(eg.pos_positions(window=1))
        # print pos
        # raise Exception()    
        return (data,labels)

    def train(self,egs):
        # Trains a classifier for each word sense
        data,labels = self.prepare_examples(egs,verbose=True)
        print "\nTraining on %d words"%len(data),
        for word in labels.iterkeys():
            sys.stdout.write(".")
            sys.stdout.flush()
            # Extract features
            self.vectorizers[word] = Vectorizer()
            X = self.vectorizers[word].fit_transform(data[word])
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
            data,labels = self.prepare_examples([eg])
            X = self.vectorizers[eg.word].transform(data[eg.word])
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
    classifier = scikit_classifier()
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

    

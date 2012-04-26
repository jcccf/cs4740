import MVectorizer
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import LabelBinarizer, Normalizer
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import RidgeClassifier
from sklearn.svm.sparse import LinearSVC
# from sklearn.linear_model.sparse import SGDClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.utils.extmath import density
# from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from NGramModel import NGramModel

import scipy.sparse as sps # sps.csr_matrix, sps.hstack
import numpy

class scikit_classifier(object):
    def __init__(self,feature_vectorizer,label_vectorizer,classifier):
        self.feature_vectorizer = feature_vectorizer
        self.label_vectorizer = label_vectorizer
        self.inverse_label_vectorizer = dict([ (v,k) for k,v in self.label_vectorizer.iteritems() ])
        self.classifier = classifier
    
    def batch_classify(self, featuresets):
        X = self.feature_vectorizer.transform(featuresets)
        X = Normalizer().fit_transform(X)
        # print X.shape
        y = self.classifier.predict(X)
        return [self.inverse_label_vectorizer[cls] for cls in y]
        
    def classify(self, featureset):
        X = self.feature_vectorizer.transform([featureset])
        X = Normalizer().fit_transform(X)
        y = self.classifier.predict(X)
        # print X.shape,y.shape
        assert(len(y) == 1)
        return self.inverse_label_vectorizer[y[0]]
        
    def prob_classify(self, featureset):
        raise NotImplementedException()
        
    @staticmethod
    def train(labeled_featuresets, C=1e5):
        """
        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        """
        feat = [featureset for featureset, label in labeled_featuresets]
        feature_vectorizer = MVectorizer.DictsVectorizer()
        X = feature_vectorizer.fit_transform(feat)
        X = Normalizer().fit_transform(X)
        label_set = set( [label for featureset, label in labeled_featuresets] )
        label_vectorizer = dict( [(label,num) for num,label in enumerate(label_set)] )
        y = numpy.array([label_vectorizer[label] for featureset, label in labeled_featuresets])
        print "Training on %d examples with %d features..."%(X.shape[0],X.shape[1]),
        classifier = OneVsRestClassifier(LinearSVC(loss='l2', penalty='l2', dual=True, tol=1e-5, C=C, scale_C=True))
        classifier.fit(X,y)
        print "done"

        return scikit_classifier(feature_vectorizer,label_vectorizer,classifier)

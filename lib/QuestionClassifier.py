# get training set 5 and test set from http://cogcomp.cs.illinois.edu/Data/QA/QC/
# put train_5500.label and TREC_10.label in data/train/qc

import nltk
import re
import os, cPickle as pickle
from pprint import pprint
from naivebayes import NaiveBayesClassifier
from scikit_classifier import scikit_classifier
from collections import defaultdict
from keywords_pickle_builder import load_keywords

keywordlist = ['do','substance','currency','religion','instrument','last','other','code','num','ord','speed','time','weight','body','def','desc','quot','state','abb','dimen','plant','popu','group','title','mount','dise','job','act','prod','art','vessel','food','anim','abb','term','city','comp','country','date','eff','dist','event','lang','loca','money','name','nick','peop','perc','sport','prof','temp','title','univ','vessel','eff','cause','tech','letter','symbol','word','color','big','fast','invent','discover','live','wrote','born']

#converts question categories in Li Roth Taxonomy to wordnet
def liroth_to_wordnet(self, category):
    pass
    #to do  
class QuestionClassifier:
    def __init__(self, fine_grain=True):
        self.keywordlist2 = load_keywords()
        train_set = self.load_labelled_data('data/train/qc/train_5500.label', fine_grain=fine_grain)
        self.train_classifier(train_set, fine_grain=fine_grain)
    
    # This is not the right way to define the features...
    # def question_features(self, question):
        # features = {}
        # words = nltk.word_tokenize(question)
        # features['words'] = ' '.join(words[0:2])
        # keywords_occured = []
        # for keyword in keywordlist:
            # if len(re.findall(' '+keyword, question)) > 0:
                # keywords_occured.append(keyword)
        # features['keywords'] = ' '.join(keywords_occured)
        # keywords_occured2 = []
        # for word in words:
            # if word in self.keywordlist2:
                # keywords_occured2.append(self.keywordlist2[word])
        # features['keywords2'] = ' '.join(keywords_occured2)
        # return features
        
    def question_features(self, question):
        features = defaultdict(int)
        words = nltk.word_tokenize(question)
        # First few words (positional)
        for idx,w in enumerate(words[0:2]):
            features['w%d:'%idx + w.lower()] = 1
        # Bag of words of keywords
        for word in words:
            if word in self.keywordlist2:
                features['kw:'+self.keywordlist2[word]] += 1
        return features
        
    def load_labelled_data(self, filename, fine_grain=True):
        train_set = []
        with open(filename, 'r') as f:
            for line in f:
                if fine_grain:
                    match = re.match('([A-Z]+:[a-z]+) (.+)', line)
                else:
                    match = re.match('([A-Z]+):[a-z]+ (.+)', line)
                train_set.append((self.question_features(match.groups()[1]),match.groups()[0]))
        return train_set
    
    def train_classifier(self, train_set, C=1e5, fine_grain=True):
        # self.classifier = NaiveBayesClassifier.train(train_set)
        # self.classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
        # self.classifier = nltk.MaxentClassifier.train(train_set,algorithm="cg",sparse=True,max_iter=50,trace=3)
        self.classifier = scikit_classifier.train(train_set,C=C)
    
    def classify(self, question):
        return self.classifier.classify(self.question_features(question))

if __name__ == '__main__':
    fine_grain = True
    classifier = QuestionClassifier(fine_grain)
    test_set = classifier.load_labelled_data('data/train/qc/TREC_10.label',fine_grain)
    train_set = classifier.load_labelled_data('data/train/qc/train_5500.label', fine_grain=fine_grain)
    for C in [10**x for x in range(0,8)]:
    # for C in [0.1,0.01,0.001,0.0001,0.00001]:
        classifier.train_classifier(train_set, C=C, fine_grain=fine_grain)
        print C,":",nltk.classify.accuracy(classifier.classifier, test_set)
    classifier.train_classifier(train_set, fine_grain=fine_grain)
    print classifier.classify('How much money does the Sultan of Brunei have?')
    print classifier.classify('When did Geraldine Ferraro run for vice president?')
    print classifier.classify('What is the nickname of Pennsylvania?') #it got this wrong
    print classifier.classify('Who is Desmond Tutu?')
    print classifier.classify('How fast can a Corvette go?')

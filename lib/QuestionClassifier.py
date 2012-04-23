# get training set 5 and test set from http://cogcomp.cs.illinois.edu/Data/QA/QC/
# put train_5500.label and TREC_10.label in data/train/qc

import nltk
import re

keywordlist = ['do','substance','currency','religion','instrument','last','other','code','num','ord','speed','time','weight','body','def','desc','quot','state','abb','dimen','plant','popu','group','title','mount','dise','job','act','prod','art','vessel','food','anim','abb','term','city','comp','country','date','eff','dist','event','lang','loca','money','name','nick','peop','perc','sport','prof','temp','title','univ','vessel','eff','cause','tech','letter','symbol','word','color','big','fast','invent','discover','live','wrote','born']

def question_features(question):
    features = {}
    words = nltk.word_tokenize(question)
    features['words'] = words[0]+words[1]
    keywords_occured = ''
    for keyword in keywordlist:
        if len(re.findall(' '+keyword, question)) > 0:
            keywords_occured += keyword
    features['keywords'] = keywords_occured
    return features

def train_classifier():
    train_set = []
    with open('data/train/qc/train_5500.label', 'r') as f:
        for line in f:
            #match = re.match('([A-Z]+):[a-z]+ (.+)', line)
            match = re.match('([A-Z]+:[a-z]+) (.+)', line)
            train_set.append((question_features(match.groups()[1]),match.groups()[0]))
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier

class QuestionClassifier:
  def __init__(self):
    self.classifier = train_classifier()
  
  def classify(self, question):
    return self.classifier.classify(question_features(question))

if __name__ == '__main__':
    classifier = QuestionClassifier()
    test_set = []
    with open('data/train/qc/TREC_10.label', 'r') as f2:
        for line in f2:
            #match = re.match('([A-Z]+):[a-z]+ (.+)', line)
            match = re.match('([A-Z]+:[a-z]+) (.+)', line)
            test_set.append((question_features(match.groups()[1]),match.groups()[0]))
    print nltk.classify.accuracy(classifier.classifier, test_set)
    print classifier.classify('How much money does the Sultan of Brunei have?')
    print classifier.classify('When did Geraldine Ferraro run for vice president?')
    print classifier.classify('What is the nickname of Pennsylvania?') #it got this wrong
    print classifier.classify('Who is Desmond Tutu?')
    print classifier.classify('How fast can a Corvette go?')

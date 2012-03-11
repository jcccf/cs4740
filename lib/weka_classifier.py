import nltk.classify.util
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier
import Parser,sys,os, string
from nltk.classify.weka import ARFF_Formatter

class weka_classifier:
    def __init__(self,window_size=-1,
            classifier=nltk.NaiveBayesClassifier,
            stopwords=nltk.corpus.stopwords.words('english'),
            split_pre_post=True,     # Splits the words before and after into two feature sets
            normalize_vec=False,     # Normalizes each feature vector to unit (L1) vectors
            remove_infrequent=0.1,   # Removes infrequent words occuring in less than 0.1 percent of examples
            pos_tags=False           # Use NLTK's part-of-speech tagger for context words
            ):
        self.classifier = classifier
        self.window_size = window_size
        self.word_senses = dict()
        self.egs = dict()
        self.classifiers = dict()  #trained models
        self.stopwords = stopwords
        self.split_pre_post = split_pre_post
        self.normalize_vec = normalize_vec
        self.remove_infrequent = remove_infrequent
        self.pos_tags = pos_tags
        print self.classifier

    def __tovec(self, d, context, prefix):
        # Word list to dictoary (sparse vector) converter
        for w in context:
            ww = prefix+w
            if ww in d:
                d[ww] += 1
            else:
                d[ww] = 1
        return d

    def __stopwords_filter(self, words):
        # Filters out words in the stopwords list
        return [ w for w in words if not w in self.stopwords and not w in string.punctuation ]

    def __remove_words_in_dict(self, d, words):
        # Remove <words> from dictionary d
        for w in words:
            if w in d:
                del d[w]
        return d

    def filter_infrequent(self, fvec, limit=0.1):
        # Filters out words that do not occur in more than <limit> of total examples
        d = dict()
        for f, label in fvec:
#            label = int(label)  # 0 or 1
            for w in f.iterkeys():
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
        limit = float(limit) * len(fvec)
        removed_words = [ w for w,l in d.iteritems() if l < limit ]
        # print len(removed_words),"out of", len(d)
        new_fvec = [ (self.__remove_words_in_dict(f,removed_words), lab) for (f,lab) in fvec ]
        return new_fvec

    def __build_eg(self, eg):
        # Applies features to example.  Currently, only co-occurence and 
        # target part-of-speech features are included
        d = dict()  # Sparse feature vector
        # POS feature
        d["__POS_"+eg.pos+"__"] = 1
        # Target feature
        d["__TARGET_"+eg.target+"__"] = 1
        # co-occurence features:
        pre_words = eg.context_before.lower().split()
        post_words = eg.context_after.lower().split()
        if self.stopwords != None and len(self.stopwords) != 0:
            pre_words = self.__stopwords_filter(pre_words)
            post_words = self.__stopwords_filter(post_words)
        if self.window_size != -1:
            pre_words = pre_words[-self.window_size:]
            post_words = post_words[:self.window_size]
        if self.split_pre_post:
            self.__tovec(d, pre_words, "-")
            self.__tovec(d, post_words, "+")
        else:
            self.__tovec(d, pre_words, "")
            self.__tovec(d, post_words, "")
        if self.pos_tags:
            d = dict(d.items() + eg.pos_positions())
        return d

    def prepare_examples(self,egs,use_str=False):
        # Prepares the examples into training data, applying features etc.
        print "Preparing on %d examples"%len(egs),
        for eg in egs:
            sys.stdout.write(".")
            sys.stdout.flush()
            eg.word = eg.word.lower()
            d = self.__build_eg(eg)
            if not eg.word in self.word_senses:
                self.word_senses[eg.word] = len(eg.senses)
            for idx, val in enumerate(eg.senses):
                if (eg.word,idx) in self.egs:
                    self.egs[(eg.word, idx)].append( (d, str(val) if use_str else val) )
                else:
                    self.egs[(eg.word, idx)] = [ (d,str(val) if use_str else val) ]

    def write_arff(self,save_dir):
        # Writes the data into (dense) ARFF files for use in WEKA
        for key,val in self.egs.iteritems():
            fmt = ARFF_Formatter.from_train(val)
            word,idx = key
            fname = save_dir + "%s_%d.arff"%(word,idx)
            fmt.write(fname, val)

    def normalize_val(self, val):
      val_new = []
      for d, label in val:
        sum_total = sum(d.values())
        val_new.append(( dict( [(k,float(v)/sum_total) for (k, v) in d.iteritems()] ), label))
      return val_new

    def train(self,egs):
        # Trains a classifier for each word sense
        self.prepare_examples(egs)
        print "\nTraining on %d wordsenses"%len(self.egs),
        for key,val in self.egs.iteritems():
            sys.stdout.write(".")
            sys.stdout.flush()
            val = self.filter_infrequent(val, self.remove_infrequent) if self.remove_infrequent > 0 else val
            val = self.normalize_val(val) if self.normalize_vec else val
            if self.classifier == nltk.ConditionalExponentialClassifier:
                self.classifiers[key] = self.classifier.train(val, sparse=True, trace=0, max_iter=10)
            else:
                self.classifiers[key] = self.classifier.train(val)

        print "\nDone"
        # Clear memory
        self.egs = None

    def predict(self, egs):
        # Given a list of examples, predict their word senses
        res = []
        for eg in egs:
            eg.word = eg.word.lower()
            d = self.__build_eg(eg)
#            print d
            n = self.word_senses[eg.word]
            for idx in range(0,n):
#                label = self.classifiers[ (eg.word,idx) ].prob_classify(d).prob("1")
                label = self.classifiers[ (eg.word,idx) ].classify(d)
                res.append(label)
        return res


if __name__ == '__main__':
    # classifier = weka_classifier(10,nltk.DecisionTreeClassifier)  # Does not work with sparse features
    # classifier = weka_classifier(10,nltk.ConditionalExponentialClassifier,split_pre_post=False)
    classifier = weka_classifier(10,nltk.NaiveBayesClassifier,split_pre_post=False)
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
        prec = tp/(tp+fp)
        rec  = tp/(tp+fn)
        f1   = 2.0*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
        print "%d, %d, %d, %d\n%f, %f, %f"%(tp,tn,fp,fn,prec,rec,f1)
    
    if False:
        classifier.prepare_examples(egs,use_str=True)
        try:
            os.makedirs('data/weka/')
        except:
            pass
        classifier.write_arff("data/weka/")

    

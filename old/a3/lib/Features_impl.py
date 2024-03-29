import cPickle as pickle
from collections import Counter

# Make cache directories if they don't exist
try:
  os.makedirs('data/features/')
except:
  pass

class PrefixSuffixExtractor():
    # Finds the most frequent prefixes and suffixes in a training set
    
    def __init__(self,lb=2,ub=4):
        # lb : minimum length of prefix/suffix
        # ub : 1+maximum length of prefix/suffix
        self.pre = dict() # Counts of prefixes
        self.suf = dict() # Counts of suffixes
        self.lb = lb
        self.ub = ub
        
    def inc(self,d,w):
        # increments count
        if w not in d:
            d[w] = 1
        else:
            d[w] += 1
            
    def train(self,data):
        # data = list of [list of (POS,word) tuples]
        for sequence in data:
            for pos,word in sequence[1:]:
                for nfix in range( self.lb, min(self.ub, len(word)) ):
                    self.inc(self.pre, word[:nfix])
                    self.inc(self.suf, word[-nfix:])
        return self
                    
    def get(self,ntop=1000):
        # Returns top <ntop> prefixes and suffixes as lists
        pre = sorted(self.pre.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        suf = sorted(self.suf.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        pre = [ w for w,c in pre[:ntop] ]
        suf = [ w for w,c in suf[:ntop] ]
        return (pre,suf,self.lb,self.ub)

    def show(self,ntop=1000):
        # Prints the most common prefixes and suffixes
        pre = sorted(self.pre.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        suf = sorted(self.suf.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        for (w,c),(ww,cc) in zip(pre[:ntop],suf[:ntop]):
            print w,c,ww,cc

class PrefixSuffixFeature():
    # Computes the feature vector for prefixes and suffixes
    # Basically, if the word matches a commonly seen prefix/suffix
    # then the corresponding coordinate of the vector will be set to 1.
    def __init__(self,cache_file='data/features/prefix_suffix.dat'):
        try:
            with open(cache_file,'r') as f:
                pre,suf,lb,ub = pickle.load(f)
            self.pre = dict( zip(pre,range(len(pre))) )
            self.suf = dict( zip(suf,range(len(suf))) )
            self.lb = lb
            self.ub = ub
        except:
            print "Error: Feature files not generated.  Try running \"Features.py -g\""
    def transform(self,observations,position,window_pos):
        # Actual function called to compute
        # Vector is represented as a sparse vector with
        # a lit of (index,value) tuples
        word = observations[position]
        f = []
        for nfix in range( self.lb, min(self.ub, len(word)) ):
            if word[:nfix] in self.pre:
                f.append( (self.pre[ word[:nfix] ], 1) )
            if word[-nfix:] in self.suf:
                f.append( (len(self.pre)+self.suf[ word[-nfix:] ], 1) )
        return f
    # Length functions return the length of the full feature vector
    def len(self):
        return len(self.pre)+len(self.suf)
    def __len__(self):
        return self.len()

class WordExtractor():
    # Extracts the vocabulary of the training set
    
    def __init__(self):
        self.vocab = set()
    
    def train(self,data):
        # data = list of [list of (POS,word) tuples]
        for sequence in data:
            for pos,word in sequence[1:]:
                self.vocab.add(word)
        return self
    
    def get(self):
        return list(self.vocab)

class POSExtractor():
    # Extracts the set of parts-of-speech tags used
    
    def __init__(self):
        self.vocab = set()
    
    def train(self,data):
        # data = list of [list of (POS,word) tuples]
        for sequence in data:
            for pos,word in sequence[1:]:
                self.vocab.add(pos)
        return self
    
    def get(self):
        return [ pos for pos in self.vocab ]

class WordFeature():
    # Extracts word feature.  Loads the pre-computed vocabulary
    # and if a given word matches a vocabulary word, the corresponding
    # coordinate of the feature vector is set to 1
    
    def __init__(self,cache_file='data/features/words.dat'):
        try:
            with open(cache_file,'r') as f:
                words = pickle.load(f)
            self.words = dict( zip(words,range(len(words))) )
        except:
            print "Error: Feature files not generated.  Try running \"Features.py -g\""
    def transform(self,observations,position,window_pos):
        # Actual computation of sparse feature vector
        word = observations[position]
        if word in self.words:
            return [ (self.words[word],1) ]
        else:
            return []
    # Length functions return the length of the full feature vector
    def len(self):
        return len(self.words)
    def __len__(self):
        return self.len()
        
class WordLengthFeature():
    def __init__(self,cache_file='data/features/words.dat'):
        try:
            with open(cache_file, 'r') as f:
                words = pickle.load(f)
            self.maxlen = float(max([len(word) for word in words]))
        except:
            print "Error: Feature files not generated.  Try running \"Features.py -g\""
    def transform(self, observations, position, window_pos):
        word = observations[position]
        return [(0, len(word)/self.maxlen)]
    def len(self):
        return 1
    def __len__(self):
        return self.len()
  
class LetterFrequencyFeature():
    def __init__(self):
        pass
    def transform(self, observations, position, window_pos):
        word = [w for w in observations[position].lower() if w in "abcdefghijklmnopqrstuvwxyz"]
        wordlen = float(len(word))
        c = Counter(word)
        return [(ord(w)-ord('a'), v/wordlen) for w, v in c.iteritems()]
    def len(self):
        return 26
    def __len__(self):
        return self.len()
    
class SentenceLengthFeature():
    def __init__(self):
        self.maxlen = 60.0 # HARDCODED VALUE
    def transform(self, observations, position, window_pos):
        if window_pos == 0:
            return [(0,min(60,len(observations)/self.maxlen))]
        else:
            return []
    def len(self):
        return 1
    def __len__(self):
        return self.len()
  
class CapitalizedFeature():
    # Extracts features of whether the first char of a word is 
    # capitalized, and whether a word is all-capitals
    def __init__(self):
        self.caps = set([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    def transform(self,observations,position, window_pos):
        # Computes sparse feature vector
        word = observations[position]
        f = []
        if word[0] in self.caps:
            f.append( (0,1) ) # Starts with a capital
            if all( [ w in self.caps for w in word[1:] ] ):
                f.append( (1,1) ) # All capitals
        return f
    # Length functions return the length of the full feature vector
    def len(self):
        return 2
    def __len__(self):
        return self.len()

class NumberFeature():
    # Extracts features of whether the word is numeric
    def __init__(self):
        self.chars = set([c for c in "0123456789."])
        self.nums = set([c for c in "0123456789"])
    def transform(self,observations,position, window_pos):
        # Computes sparse feature vector
        word = observations[position]
        f = []
        if all( [ c in self.chars for c in word ] ) \
            and any( [ c in self.nums for c in word ] ):
            f.append( (0,1) )
        return f
    # Length functions return the length of the full feature vector
    def len(self):
        return 1
    def __len__(self):
        return self.len()
    
class PunctuationFeature():
    # Extracts the kinds of punctuation in the sentence
    def __init__(self,cache_file='data/features/words.dat'):
        try:
            with open(cache_file, 'r') as f:
                words = pickle.load(f)
            numbers_letters = set([c for c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"])
            punctuation = [ w for w in words if not any( [c for c in w if c in numbers_letters] ) ]
            self.punctuation = dict( zip(punctuation,range(len(punctuation))) )
        except:
            print "Error: Feature files not generated.  Try running \"Features.py -g\""
    def transform(self,observations,position, window_pos):
        # Computes sparse feature vector
        if window_pos == 0:
            v = set()
            for word in observations:
                if word in self.punctuation:
                    v.add(self.punctuation[word])
            return [ (idx,1) for idx in sorted(v) ]
        else:
            return []
    # Length functions return the length of the full feature vector
    def len(self):
        return len(self.punctuation)
    def __len__(self):
        return self.len()
    
class FeatureVectorizer():
    # Takes a list of features, and computes them over a specified 
    # window, and returns a sparse vector of features
    # Makes sure that the indices do not overlap between different
    # sets of features
    def __init__(self,window=[-1,0,1], features=[]
                # features=[CapitalizedFeature(),WordFeature(),PrefixSuffixFeature()]
                ):
        self.features = features
        self.window = window
        self.feature_len = sum( [len(f) for f in self.features] )
        
    def transform(self,observations,position):
        # Actual computation of sparse feature vector
        idx_base = 1
        f = []
        for w in self.window:
            if position+w < 0 or position+w >= len(observations):
                idx_base += self.feature_len
            else:
                for feat in self.features:
                    g = feat.transform(observations,position+w,w)
                    f.extend( [ (i+idx_base,v) for i,v in g ] )
                    idx_base += len(feat)
        f = sorted(f, cmp=lambda (a,b),(c,d): a-c)
        return f
    
    # Length functions return the length of the full feature vector
    def len(self):
        return len(self.window)*self.feature_len

    def __len__(self):
        return self.len()

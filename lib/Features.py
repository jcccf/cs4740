import argparse,os
import cPickle as pickle
from pprint import pprint
from Parser import *
from collections import Counter

# Make cache directories if they don't exist
try:
  os.makedirs('data/features/')
except:
  pass

class PrefixSuffixExtractor():
    def __init__(self,lb=2,ub=4):
        self.pre = dict()
        self.suf = dict()
        self.lb = lb
        self.ub = ub
        
    def inc(self,d,w):
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
                    
    def get(self,ntop=100):
        # Returns top <ntop> prefixes and suffixes as lists
        pre = sorted(self.pre.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        suf = sorted(self.suf.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        pre = [ w for w,c in pre[:ntop] ]
        suf = [ w for w,c in suf[:ntop] ]
        return (pre,suf,self.lb,self.ub)

    def show(self,ntop=100):
        # Prints the most common prefixes and suffixes
        pre = sorted(self.pre.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        suf = sorted(self.suf.iteritems(), cmp=lambda (a,b),(c,d): d-b)
        for (w,c),(ww,cc) in zip(pre[:ntop],suf[:ntop]):
            print w,c,ww,cc

class PrefixSuffixFeature():
    def __init__(self,cache_file='data/features/prefix_suffix.dat'):
        with open(cache_file,'r') as f:
            pre,suf,lb,ub = pickle.load(f)
        self.pre = dict( zip(pre,range(len(pre))) )
        self.suf = dict( zip(suf,range(len(suf))) )
        self.lb = lb
        self.ub = ub
    def transform(self,observations,position):
        word = observations[position]
        f = []
        for nfix in range( self.lb, min(self.ub, len(word)) ):
            if word[:nfix] in self.pre:
                f.append( (self.pre[ word[:nfix] ], 1) )
            if word[-nfix:] in self.suf:
                f.append( (len(self.pre)+self.suf[ word[-nfix:] ], 1) )
        return f
    def len(self):
        return len(self.pre)+len(self.suf)
    def __len__(self):
        return self.len()

class WordExtractor():
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
    def __init__(self,cache_file='data/features/words.dat'):
        with open(cache_file,'r') as f:
            words = pickle.load(f)
        self.words = dict( zip(words,range(len(words))) )
    def transform(self,observations,position):
        word = observations[position]
        if word in self.words:
            return [ (self.words[word],1) ]
        else:
            return []
    def len(self):
        return len(self.words)
    def __len__(self):
        return self.len()
        
class WordLengthFeature():
  def __init__(self,cache_file='data/features/words.dat'):
    with open(cache_file, 'r') as f:
      words = pickle.load(f)
    self.maxlen = float(max([len(word) for word in words]))
  def transform(self, observations, position):
    word = observations[position]
    return [(0, len(word)/self.maxlen)]
  def len(self):
    return 1
  def __len__(self):
    return self.len()
  
class LetterFrequencyFeature():
  def __init__(self):
    pass
  def transform(self, observations, position):
    word = [w for w in observations[position].lower() if w in "abcdefghijklmnopqrstuvwxyz"]
    wordlen = float(len(word))
    c = Counter(word)
    return [(ord(w)-ord('a'), v/wordlen) for w, v in c.iteritems()]
  def len(self):
    return 26
  def __len__(self):
    return self.len()

class CapitalizedFeature():
    def __init__(self):
        self.caps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def transform(self,observations,position):
        word = observations[position]
        f = []
        if word[0] in self.caps:
            f.append( (0,1) ) # Starts with a capital
            if all( [ w in self.caps for w in word[1:] ] ):
                f.append( (1,1) ) # All capitals
        return f
    def len(self):
        return 2
    def __len__(self):
        return self.len()

class FeatureVectorizer():
    def __init__(self,window=[-1,0,1], features=[]
                # features=[CapitalizedFeature(),WordFeature(),PrefixSuffixFeature()]
                ):
        self.features = features
        self.window = window
        self.feature_len = sum( [len(f) for f in self.features] )
        
    def transform(self,observations,position):
        idx_base = 1
        f = []
        for w in self.window:
            if position+w < 0 or position+w >= len(observations):
                idx_base += self.feature_len
            else:
                for feat in self.features:
                    g = feat.transform(observations,position+w)
                    f.extend( [ (i+idx_base,v) for i,v in g ] )
                    idx_base += len(feat)
        f = sorted(f, cmp=lambda (a,b),(c,d): a-c)
        return f
    
    def len(self):
        return len(self.window)*self.feature_len

    def __len__(self):
        return self.len()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates training/test files.')
    parser.add_argument('-n', metavar='N_TOP', type=int, dest='ntop', default=1000, 
                       help='number of top prefix/suffix to use')
    parser.add_argument('-g', dest='generate', action='store_true',
                       help='Whether to generate feature set(s)')
    parser.add_argument('-d', type=str, metavar='DIR', dest='outdir', default='data/features/',
                       help='Where to save/load pickled data')
    parser.add_argument('--train', type=argparse.FileType('r'), metavar='FILE', dest='trainfile',
                       # default='data/pos_files/train.pos',
                       default=None,
                       help='Where to get training data')
    parser.add_argument('--test', type=argparse.FileType('r'), metavar='FILE', dest='testfile',
                       default='data/pos_files/test-obs.pos',
                       # default=None,
                       help='Where to get test data')
    
    args = parser.parse_args()
    pprint(args)
    
    if args.generate:
        data = parse_training_file()
        datalen = float(len(data))
        # fix_extractor = PrefixSuffixExtractor()
        # fix_extractor.train(data)
        feat = PrefixSuffixExtractor().train(data).get(ntop=args.ntop)
        with open(args.outdir+"prefix_suffix.dat","w") as f:
            pickle.dump(feat,f)
        feat = WordExtractor().train(data).get()
        with open(args.outdir+"words.dat","w") as f:
            pickle.dump(feat,f)
        feat = POSExtractor().train(data).get()
        with open(args.outdir+"pos.dat","w") as f:
            pickle.dump(feat,f)
    else:
        with open(args.outdir+"pos.dat",'r') as f:
            pos = pickle.load(f)
        POS_to_idx = dict( zip(pos,range(1,1+len(pos))) )
        
        fv = FeatureVectorizer(features=[CapitalizedFeature(),WordFeature(),PrefixSuffixFeature(),WordLengthFeature(),LetterFrequencyFeature()])
        # fv = FeatureVectorizer(features=[WordFeature()])
        print "Total number of features:",fv.len()
        # exit(0)
        if args.trainfile != None:
            training_data = parse_opened_training_file(args.trainfile)
            with open(args.trainfile.name + ".features",'w') as output_file:
                for qid,sequence in enumerate(training_data):
                    qid = qid+1
                    words = [w for p,w in sequence[1:]]
                    tags =  [POS_to_idx[p] for p,w in sequence[1:]]
                    for position in range(0,len(words)):
                        g = fv.transform(words,position)
                        tag = tags[position]
                        line = "%d qid:%d "%(tag,qid)
                        output_file.write(line)
                        line = " ".join( ["%d:%g"%(i,v) for i,v in g] )
                        output_file.write(line)
                        line = " # %s\n"%words[position]
                        output_file.write(line)
        if args.testfile != None:
            training_data = parse_opened_test_file(args.testfile)
            with open(args.testfile.name + ".features",'w') as output_file:
                tag = 1
                for qid,sequence in enumerate(training_data):
                    qid = qid+1
                    words = sequence[1:]
                    for position in range(0,len(words)):
                        g = fv.transform(words,position)
                        line = "%d qid:%d "%(tag,qid)
                        output_file.write(line)
                        line = " ".join( ["%d:%g"%(i,v) for i,v in g] )
                        output_file.write(line)
                        line = " # %s\n"%words[position]
                        output_file.write(line)
    
    
    
    
    
    
    
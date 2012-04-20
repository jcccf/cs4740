# from scipy.sparse import csr_matrix
import math,random,argparse,time,inspect
from pprint import pprint
from cProfile import run
from Parser import *

class HMM():
    def __init__(self, ngram=2, smooth="lap"):
        self.tp = dict() # Transition probabilities
        self.ep = dict() # Emission probabilities
        self.vocab = set() # Set of observations
        self.pos = set()   # Set of states
        self.ngram = ngram
        if smooth == "lap":
            self.smooth = HMM.laplacian_smoothing
        elif smooth == "frac":
            self.smooth = HMM.fractional_smoothing
        elif smooth == "none":
            self.smooth = HMM.no_smoothing
        elif inspect.isfunction(smooth):
            self.smooth = smooth
        else:
            self.smooth = HMM.no_smoothing
    
    def add_count(self,prob,curr,word):
        # Adds transition/emission probability count
        if curr not in prob:
            prob[curr] = dict()
        d = prob[curr]
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    
    @staticmethod
    def laplacian_smoothing(prob,vocab):
        # Applies add-one smoothing, and normalizes counts
        # to log-probabilities
        # Note: special key -1 used to catch all cases that
        # were not seen in the training set, since our
        # representation is sparse
        vocabsize = len(vocab)
        for d in prob.itervalues():
            total = sum(d.itervalues())
            denom = math.log(total+vocabsize)
            for key,val in d.iteritems():
                d[key] = math.log(val + 1) - denom
            d[-1] = -denom
        prob[-1] = -math.log(vocabsize)
                
    @staticmethod
    def fractional_smoothing(prob,vocab,fraction=0.05):
        # Applies smoothing where none-transitions occupy
        # <fraction> probability.
        # Normalizes counts to log-probabilities
        # Note: special key -1 used to catch all cases that
        # were not seen in the training set, since our
        # representation is sparse
        vocabsize = len(vocab)
        p = float(fraction)
        omp = 1.0-p
        for d in prob.itervalues():
            total = sum(d.itervalues())
            for key,val in d.iteritems():
                val = omp*val/total + p/vocabsize
                d[key] = math.log(val)
            d[-1] = math.log( p/vocabsize )
        prob[-1] = -math.log(vocabsize)
        
    @staticmethod
    def no_smoothing(prob,vocab):
        # Just normalizes counts
        # Note: special key -1 used to catch all cases that
        # were not seen in the training set, since our
        # representation is sparse
        for d in prob.itervalues():
            total = math.log(sum(d.itervalues()))
            for key,val in d.iteritems():
                d[key] = math.log(val) - total
            d[-1] = math.log(1e-8)
        prob[-1] = math.log(1e-8)
        
    @staticmethod
    def get_log_probability(prob,curr,word):
        # Returns log( Pr[ word | curr ] )
        if curr not in prob:
            return prob[-1] # Special catch-all case
        d = prob[curr]
        if word in d:
            return d[word]
        else:
            return d[-1] # Special catch-all case
    
    def train(self, data):
        # data = list of [list of (POS,word) tuples]
        for sequence in data:
            prev_state = ("<s>",)*(self.ngram-1)
            for pos,word in sequence[1:]:
                self.vocab.add(word)
                self.pos.add(pos)
                curr_state = prev_state[1:] + (pos,)
                self.add_count(self.tp, prev_state, pos)
                self.add_count(self.ep, curr_state, word)
                prev_state = curr_state
        # Normalize counts with smoothing
        self.smooth(self.tp, self.pos)
        self.smooth(self.ep, self.vocab)
        # pprint(self.tp)
        # pprint(self.ep)
        
    def decode(self, observations):
        # Uses Viterbi to infer the most likely state-sequence generating the
        # observation-sequence. Guaranteed to find the most likely sequence, at
        # the cost of O( T L^ngram ) time complexity.
        #   observations : list of words
        #   returns (most-likely-states, log-probability of states)
        V = dict()
        path = dict()
        prev_state = ("<s>",)*(self.ngram-1)
        word = observations[1]
        # Initialize beginning
        for pos in self.pos:
            curr_state = prev_state[1:] + (pos,)
            V[curr_state] = ( HMM.get_log_probability(self.tp,prev_state,pos) 
                            + HMM.get_log_probability(self.ep,curr_state,word) )
            path[curr_state] = ("<s>",pos)
        # Compute for subsequent words
        for word in observations[2:]:
            Vnew = dict()
            Pnew = dict()
            for pos in self.pos:
                # state,prob = None,-float("inf")
                for prev_state,val in V.iteritems():
                    curr_state = prev_state[1:] + (pos,)
                    val += ( HMM.get_log_probability(self.tp,prev_state,pos)
                           + HMM.get_log_probability(self.ep,curr_state,word) )
                    if (curr_state not in Vnew) or (Vnew[curr_state] < val):
                        Vnew[curr_state] = val
                        Pnew[curr_state] = path[prev_state] + (pos,)
            V = Vnew
            path = Pnew
        prob,state = max( [ (val,key) for key,val in V.iteritems() ] )
        return (path[state],prob)
        
    def decode_fast(self, observations):
        # Uses Viterbi to infer the most likely state-sequence generating the
        # observation-sequence.  This method is not guaranteed to find the best
        # sequence for ngram > 2, but performs much faster O( T L^2 ).
        #   observations : list of words
        #   returns (most-likely-states, log-probability of states)
        V = dict()
        path = dict()
        prev_state = ("<s>",)*(self.ngram-1)
        word = observations[1]
        # Initialize beginning
        for pos in self.pos:
            curr_state = prev_state[1:] + (pos,)
            V[curr_state] = ( HMM.get_log_probability(self.tp,prev_state,pos) 
                            + HMM.get_log_probability(self.ep,curr_state,word) )
            path[pos] = ("<s>",pos)
        # Compute for subsequent words
        for word in observations[2:]:
            Vnew = dict()
            Pnew = dict()
            for pos in self.pos:
                state,prob = None,-float("inf")
                for prev_state,val in V.iteritems():
                    curr_state = prev_state[1:] + (pos,)
                    val += ( HMM.get_log_probability(self.tp,prev_state,pos)
                           + HMM.get_log_probability(self.ep,curr_state,word) )
                    if val > prob:
                        state = prev_state
                        prob = val
                curr_state = state[1:] + (pos,)
                prev_pos = state[-1]
                Vnew[curr_state] = prob
                Pnew[pos] = path[prev_pos] + (pos,)
                # print Pnew[pos],prob
            V = Vnew
            path = Pnew
        prob,state = max( [ (val,key) for key,val in V.iteritems() ] )
        pos = state[-1]
        return (path[pos],prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HMM.')
    parser.add_argument('-n', metavar='ngram', type=int, dest='ngram', default=2, 
                       help='ngram for HMM')
    parser.add_argument('-k', metavar='kfold', type=int, dest='kfold', default=5, 
                       help='# folds for cross validation')
    parser.add_argument('-s', metavar='random_seed', type=int, dest='random_seed', default=1023, 
                       help='seed for random number generator')

    args = parser.parse_args()
    pprint(args)
    
    kfold = args.kfold
    # random.seed(args.random_seed)
    # data = [ zip("<s> I LIKE TO EAT ICE CREAM".split(),
                 # "<s> i like to eat ice cream".split()),
             # zip("<s> DOGS LIKE TO EAT ICE CREAM".split(),
                 # "<s> DOGS like to eat ice cream".split()) ]
    # print data
    # teststr = "<s> The plant , which is owned by Hollingsworth & Vose Co. , was under contract with Lorillard to make the cigarette filters .".split()
    
    data = parse_training_file()
    datalen = float(len(data))
    # k-fold cross validation
    # total = 0.0
    # correct = 0.0
    # print "Shuffling data...",
    # random.shuffle(data)
    # print "Done."
    # for i in range(kfold):
        # print "Fold %d: Spliting data..."%(i+1),
        # lb = int(math.ceil(i*datalen/kfold))
        # ub = len(data) if i==5 else int(math.ceil((i+1)*datalen/kfold))
        # training_data = data[0:lb] + data[ub:]
        # test_data = data[lb:ub]
        # print "Training...",
        # hmm = HMM(ngram=args.ngram)
        # hmm.train(training_data)
        # print "Testing...",
        # for eg in test_data:
            # eg_pos = [ pos for pos,word in eg ]
            # teststr = [ word for pos,word in eg ]
            # seq,prob = hmm.decode(teststr)
            # correct += sum([1.0 for a,b in zip(seq,eg_pos) if a==b])
            # total += len(eg_pos)
        # print "Done"
    # print "Correct:",correct
    # print "Total:",total
    # print "Accuracy:",correct/total
    
    lb = 123
    ub = 150
    training_data = data[0:lb] + data[ub:]
    test_data = data[lb:ub]
    
    t = time.time()
    for n in range(2,4):
        # for smooth in ["lap","frac","none"]:
        for smooth in [lambda x,y: HMM.fractional_smoothing(x,y,fraction=0.99)]:
            print n,
            print smooth,
            accuracy = 0
            random.seed(args.random_seed)
            for n_egs in range(10):
                eg = random.choice(test_data)
                eg_pos = [ pos for pos,word in eg ]
                teststr = [ word for pos,word in eg ]
                hmm = HMM(ngram=n,smooth=smooth)
                hmm.train(training_data)
                # seq,prob = hmm.decode(teststr)
                seq,prob = hmm.decode_fast(teststr)
                accuracy += sum([1 for a,b in zip(seq,eg_pos) if a==b]) / float(len(eg_pos))
                # print prob, " ".join(seq),
                # print accuracy
            print accuracy / 10.0
    print "Took",time.time()-t,"secs."
    # hmm = HMM(ngram=3)
    # hmm.train(data)
    # seq,prob = hmm.decode(teststr)
    # print sum([1.0 for a,b in zip(seq,eg_pos) if a==b]) / len(eg_pos)
    
    
    
    
    
    
    
    
    
    
    
    
    
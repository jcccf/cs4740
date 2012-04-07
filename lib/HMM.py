# from scipy.sparse import csr_matrix
import math,random,argparse
from pprint import pprint
from cProfile import run

# CNT = 0

class HMM():
    def __init__(self, ngram=2, smooth="lap"):
        self.tp = dict()
        self.ep = dict()
        self.vocab = set()
        self.pos = set()
        self.ngram = ngram
        if smooth == "lap":
            self.smooth = HMM.laplacian_smoothing
    
    def add_tp_count(self,prev,pos):
        if prev not in self.tp:
            self.tp[prev] = dict()
        d = self.tp[prev]
        if pos not in d:
            d[pos] = 1
        else:
            d[pos] += 1
            
    def add_ep_count(self,curr,word):
        if curr not in self.ep:
            self.ep[curr] = dict()
        d = self.ep[curr]
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    
    @staticmethod
    def laplacian_smoothing(prob,vocab):
        vocabsize = len(vocab)
        for d in prob.itervalues():
            total = sum(d.itervalues())
            denom = math.log(total+vocabsize)
            for key,val in d.iteritems():
                d[key] = math.log(val + 1) - denom
            d[-1] = -denom
        prob[-1] = -math.log(vocabsize)
                
    @staticmethod
    def no_smoothing(prob,vocab):
        for d in prob.itervalues():
            total = math.log(sum(d.itervalues()))
            for key,val in d.iteritems():
                d[key] = math.log(val) - total
            d[-1] = math.log(1e-8)
        prob[-1] = math.log(1e-8)
        
    @staticmethod
    def get_log_probability(prob,curr,word):
        # global CNT
        # CNT += 1
        if curr not in prob:
            return prob[-1]
        d = prob[curr]
        if word in d:
            return d[word]
        else:
            return d[-1]
    
    def train(self, data):
        # data = list of [list of (POS,word) tuples]
        for sequence in data:
            prev_state = ("<s>",)*(self.ngram-1)
            for pos,word in sequence[1:]:
                self.vocab.add(word)
                self.pos.add(pos)
                curr_state = prev_state[1:] + (pos,)
                self.add_tp_count(prev_state,pos)
                self.add_ep_count(curr_state,word)
                prev_state = curr_state
        # Normalize counts with smoothing
        self.smooth(self.tp,self.pos)
        self.smooth(self.ep,self.vocab)
        # pprint(self.tp)
        # pprint(self.ep)
        
    def decode(self, observations):
        # Uses Viterbi to infer the most likely state-sequence generating the
        # observation-sequence.
        # observations : list of words
        # returns (most-likely-states, log-probability of states)
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

        
def parse_training_file(filename='data/pos_files/train.pos'):
    data = []
    sentence = []
    with open(filename,'r') as f:
        for line in f:
            s = line.split()
            if len(s) != 2:
                print "ERROR:", line
                exit(-1)
            if s[0] == "<s>" and len(sentence) > 0:
                data.append(sentence)
                sentence = []
            sentence.append( tuple(s) )
    if len(sentence) > 0:
        data.append(sentence)
    return data
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HMM.')
    parser.add_argument('-n', metavar='ngram', type=int, dest='ngram', default=2, 
                       help='ngram for HMM')
    parser.add_argument('-k', metavar='kfold', type=int, dest='kfold', default=5, 
                       help='# folds for cross validation')
    parser.add_argument('-s', metavar='random_seed', type=int, dest='random_seed', default=1023, 
                       help='seed for random number generator')

    args = parser.parse_args()
    kfold = args.kfold
    random.seed(args.random_seed)
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
    
    
    eg = random.choice(data)
    eg_pos = [ pos for pos,word in eg ]
    teststr = [ word for pos,word in eg ]
    
    hmm = HMM(ngram=args.ngram)
    hmm.train(data)
    print len(hmm.pos)
    print len(hmm.vocab)
    print (len(eg_pos)-1)*((len(hmm.pos)**2)+(len(hmm.vocab))
    # print sum([1.0 for a,b in zip(seq,eg_pos) if a==b]) / len(eg_pos)
    
    # hmm = HMM(ngram=3)
    # hmm.train(data)
    # seq,prob = hmm.decode(teststr)
    # print sum([1.0 for a,b in zip(seq,eg_pos) if a==b]) / len(eg_pos)
    
    
    
    
    
    
    
    
    
    
    
    
    
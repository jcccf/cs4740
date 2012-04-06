from random import random,randint,choice
from math import floor,log,exp
from collections import deque
import profile, string


class NGramModel():
  def __init__(self, n, smooth_type=None, unknown_type=None, gram_type="n"):
    # Options:
    #    gram_type:  "n" -> only n-tuples
    #                "all" -> all 1 to n tuples
    # self.freq:  dict( word_list => (#occurrances, dict(word=>frequency) ) )
    self.freq = dict({tuple(): (0,dict())})
    self.vocab = dict() # Explicitly store the vocab now
    self.n = n
    self.vocab_expansion = 0
    if smooth_type == None or smooth_type == 'none':
      self.smooth = self.no_smoothing
    elif smooth_type == 'lap':
      self.smooth = self.laplacian_smoothing
    elif smooth_type == 'gte':
      self.smooth = self.good_turing_smoothing
      # a table of the number of N-grams that occur c times for all values of c  
      self.ngram_count = dict()
    else:
      raise Exception("Invalid smoothing function name")
    if unknown_type == None or unknown_type == 'none':
      self.unknown = self.unknown_none
    elif unknown_type == 'first':
      self.unknown = self.unknown_first
    elif unknown_type == 'once':
      self.unknown = self.unknown_once
    else:
      raise Exception("Invalid unknown function name")
    self.gram_type = gram_type
  
  def unknown_none(self, corpus):
    return corpus
  
  def unknown_first(self, corpus):
    vocab = dict()
    new_corpus = []
    for doc in corpus:
      new_doc = []
      for word in doc:
        if not vocab.has_key(word): # First occurance
          vocab[word] = 1
          word = "<UNK>" # special type for unknown words
        new_doc.append(word)
      new_corpus.append(new_doc)
    return new_corpus
  
  def unknown_once(self, corpus):
    vocab = dict()
    new_corpus = []
    for doc in corpus:
      for word in doc:
        vocab.setdefault(word,0)
        vocab[word] += 1
    vocab_once = dict()
    for word,count in vocab.iteritems():
      if count == 1:
        vocab_once[word] = 1
    for doc in corpus:
      new_doc = []
      for word in doc:
        if vocab_once.has_key(word):
          word = "<UNK>"
        new_doc.append(word)
      new_corpus.append(new_doc)
    return new_corpus
    
  def add_ntuple(self, tup):
    head = tuple(tup[-self.n:-1])
    tail = tup[-1]
    if head in self.freq:
      (count,d) = self.freq[ head ]
      if tail in d:
        d[ tail ] += 1
      else:
        d[ tail ] = 1
      self.freq[head] = (count+1, d)
    else:
      self.freq[head] = (1, dict({tail: 1}) )

  def add_all_ntuple(self, tup):
    # tup = (w1, w2, w3, ..., wn)
    # adds (w1), (w1 w2), (w1, w2, w3) ...
    lst = []
    for w in tup:
      lst.append(w)
      self.add_ntuple(lst)
  
  def train(self, corpus):
    unk_corpus = self.unknown(corpus)
    # corpus: list of documents, each document is a list of words
    for doc in unk_corpus:
      nm1 = self.n-1
      lst = [] if self.gram_type == "all" else ["<s>"]*nm1
      for w in doc:
        self.vocab[w] = self.vocab.setdefault(w, 0) + 1
        lst = [] if self.n == 1 else lst[-nm1:]
        lst.append(w)
        assert len(lst) <= self.n
        if self.gram_type == "n":
          self.add_ntuple(lst)
        elif self.gram_type == "all":
          self.add_all_ntuple(lst)
      while (self.gram_type == "all") and len(lst) > 0: # Add trailing words
        lst = lst[1:]
        self.add_all_ntuple(lst)
    # If Good-Turing smoothing is used, adjust the counts in the model
    if self.smooth == self.good_turing_smoothing:
        self.good_turing_discount_model();
  
  def get_rand_word( self, tup ):
    if self.gram_type == "n":
      tmp = ["<s>"]*(self.n-1)
      tmp.extend(tup)
      tup = tmp
    # Given n-1 words, get the n-th word
    head = tuple(tup[ -(self.n-1): ])
    if head in self.freq:
      (count,d) = self.freq[head]
      if self.smooth == self.no_smoothing:
        #Assuming no smoothing
        pos = random() * (count)
        for (w,f) in d.iteritems():
          if pos < f:
            return w
          else:
            pos -= f
        assert False #Never should get here!
      elif self.smooth == self.laplacian_smoothing:
        # Assuming Laplacian smoothing
        pos = random() * (count + self.vocab_size())
        if pos < count:
          for (w,f) in d.iteritems():
            if pos < f:
              return w
            else:
              pos -= f
          assert False #Never should get here!
        else:
          pos = int(floor(pos-count))
          v = self.vocab_dict().keys()
          assert pos < len(v)
          return v[pos]
      elif self.smooth == self.good_turing_smoothing:
        # Assuming Good-Turing smoothing
        if isinstance(head,tuple): #a len(head)+1-gram is being processed
            gram_length = len(head)+1
        elif head == []: #a unigram is being processed
            gram_length = 1
        else: #head is a single word, hence a bigram is being processed
            gram_length = 2
        # compute the smoothed counts of possible N-grams that did not occur in the training data
        if self.ngram_count[(gram_length,0)] > 0:
            c_smoothed = float(self.ngram_count[(gram_length,1)]) / float(self.ngram_count[(gram_length,0)])
        else: # for the case of data where every possible N-gram does appear at least once
            # can't discount in this case since N0 is 0, so just add 1 to each possible N-gram
            c_smoothed = 1        
        pos = random() * (count + c_smoothed * (self.vocab_size()-len(d)))
        if pos < count:
          for (word,f) in d.iteritems():
            if pos < f:
              return word
            else:
              pos -= f
          assert False #Never should get here!
        else:
          pos = pos - count; 
          for word in self.vocab_dict().iterkeys():
            if word not in d:
              if pos < c_smoothed:
                return word
              else:
                pos -= c_smoothed
          assert False #Never should get here!                    
    else:
      pos = int(floor(random() * (self.vocab_size())))
      v = self.vocab_dict().keys()
      assert pos < len(v)
      return v[pos]
  
  def vocab_size(self):
    return len(self.vocab) + self.vocab_expansion
    # empty = tuple()
    # (count,d) = self.freq[empty]
    # return len(d)
  
  def set_vocab_expansion(self, val):
    self.vocab_expansion = val
    
  def expand_vocab(self, test):
    st = set(test)
    for w in st:
      if not w in self.vocab:
        self.vocab_expansion += 1
    
  def vocab_dict(self):
    return self.vocab
    # empty = tuple()
    # (count,d) = self.freq[empty]
    # return d
  
  def laplacian_smoothing( self, head, tail ):
    # d=freq[head]
    # count = sum(d)
    # n_unigram = length( freq[ [] ].keys() ) ## Vocabulary size
    # P(tail | head) = (#(head,tail)+1) / (#head + Vocab)
    # returns P(tail | head)
    head = tuple(head)
    if head in self.freq:
      (count,d) = self.freq[head]
      if tail in d:
        return (d[tail]+1) / float(count+self.vocab_size())
      else:
        return 1.0 / float(count+self.vocab_size())
    else:
      return 1.0 / self.vocab_size()
  
  def no_smoothing(self, head, tail):
    head = tuple(head)
    if head in self.freq:
      (count,d) = self.freq[head]
      if tail in d:
        return d[tail] / (count-0.0)
      else:
        return 0
    else:
      return 0

  def good_turing_smoothing( self, head, tail ):
    head = tuple(head)
    if head in self.freq:
        (count,d) = self.freq[head]
        if isinstance(head,tuple): #a len(head)+1-gram is being processed
            gram_length = len(head)+1
        elif head == []: #a unigram is being processed
            gram_length = 1
        else: #head is a single word, hence a bigram is being processed
            gram_length = 2
        # we have to add the smoothed count of the possible N-grams that did not occur
        # in the training data.
        # we add (0 + 1) * N1 / N0 for each of the possible N-grams that did not occur
        # (self.vocab_size()-len(d)) gives the number of N-grams beginning with the head
        # used as the key that did not appear in the test data.
        # thus we add (self.vocab_size()-len(d)) * N1 / N0 to the count
        if self.ngram_count[(gram_length,0)] > 0:
            count += (self.vocab_size()-len(d)) * float(self.ngram_count[(gram_length,1)]) / float(self.ngram_count[(gram_length,0)])
        else: # for the case of data where every possible N-gram does appear at least once
            # can't discount in this case since N0 is 0, so just add 1 to each possible N-gram
            count += (self.vocab_size()-len(d))
        if tail in d:
            # frequencies have already been smoothed by good_turing_discount_model()
            return d[tail] / float(count)
        else:
            # we compute a smoothed count for c = 0 by c_smoothed = (0+1)*N1/N0
            N0 = self.ngram_count[(gram_length,0)]
            N1 = self.ngram_count[(gram_length,1)]
            if N0 > 0:
                c_smoothed = float(N1) / float(N0)
            else: 
                # for the case of training data where every possible N-gram appears at least once
                # can't divide by N0 since N0 is 0
                c_smoothed = 1;
            # the smoothed count is then divided by the total count,
            # which was increased to take into account all the possible 
            # N-grams which did not occur, earlier in the function
            # returns the conditional probability
            return c_smoothed / float(count)
    else:
        return 1.0 / self.vocab_size()

  def good_turing_discount_model(self):  
    # we first construct table of counts of number of N-grams that occur c times for all c  
    for head in self.freq.iterkeys():
        if isinstance(head,tuple): #a len(head)+1-gram is being processed
            gram_length = len(head)+1
        elif head == []: #a unigram is being processed
            gram_length = 1
        else: #head is a single word, hence a bigram is being processed
            gram_length = 2
        (count,d) = self.freq[head]
        for (value) in d.itervalues():
            if (gram_length,value) in self.ngram_count:    
                self.ngram_count[(gram_length,value)] += 1
            else:
                self.ngram_count[(gram_length,value)] = 1
    # we must also compute counts for number of N-grams that occurred 0 times 
    for gram_length in range(1, self.n+1):
        # first set the count to be the number of possible N-grams formed from permutations of N words
        self.ngram_count[(gram_length,0)] = pow(self.vocab_size(), gram_length)      
    for (gram_length,occurrences) in self.ngram_count.iterkeys():
        # next we subtract the number of N-grams that occurred more than once
        # to get the number of N-grams that occured 0 times
        if occurrences > 0:
            self.ngram_count[(gram_length,0)] -= self.ngram_count[(gram_length,occurrences)]             
    # next we use the table of counts of number of N-grams that occur c times
    # to revise the count for each N-gram in the model, thus performing Good-Turing smoothing
    for head in self.freq.iterkeys():
        if isinstance(head,tuple): #a len(head)+1-gram is being processed
            gram_length = len(head)+1
        elif head == []: #a unigram is being processed
            gram_length = 1
        else: #head is a single word, hence a bigram is being processed
            gram_length = 2
        (count,d) = self.freq[head]
        count = 0 #count will be recomputed in the following for loop
        for tail in d.iterkeys():
            Nc = self.ngram_count[(gram_length,d[tail])]
            if (gram_length,d[tail]+1) in self.ngram_count: #at least one N-gram occured c+1 times    
                Ncplus1 = self.ngram_count[(gram_length,d[tail]+1)]
                d[tail] = float(d[tail]+1) * float(Ncplus1) / float(Nc) #adjust c to c*
            else:
                d[tail] = d[tail]+1 # can't discount if no N-grams that occur c+1 times
            count += d[tail] # reflect the adjustment in count
        self.freq[head] = (count, d) # update the model with the smoothed counts
    # smoothing complete
    
  def get_cond_prob( self, tup, unknown_substituted = False ):
    # Given n words, get log( P( wn | w1,...,wn-1 ) )
    # Given more than n words, only use the last n words and ignores the rest
    if len(tup) < 1:
      raise Exception("get_cond_prob: tup should not be an empty list")
    if not unknown_substituted:
      vd = self.vocab_dict()
      tup = map( lambda x: x if vd.has_key(x) else "<UNK>", tup )
    tup = tup[-self.n:]
    head = tup[:-1]
    tail = tup[-1]
    try:
      return log(self.smooth(head,tail))
    except:
      return float("-inf")
  
  # P( w1 w2 ... wm ) = P(w1) P(w2 | w1) P(w3 | w1, w2) ... P(wn | w1,...,wn-1) P(wn+1 | w2,...,wn) ...
  def get_prob( self, str, unknown_substituted = False ):
    # Return log of probability
    #return 0 if len(str) == 0 else self.get_prob( str[:-1] ) + self.get_cond_prob( str )
    if not unknown_substituted:
      vd = self.vocab_dict()
      str = map( lambda x: x if vd.has_key(x) else "<UNK>", str )
    acc = 0
    cur = deque([] if self.gram_type == "all" else ["<s>"]*(self.n-1), self.n)
    for w in str:
      cur.append(w)
      assert len(cur) <= self.n
      acc += self.get_cond_prob(list(cur), unknown_substituted = True)
    return acc
    
  def get_perplexity(self, str, unknown_substituted = False):
    # Perplexity = P(string)^{-1/n}, so log(perplexity) = -1/n * log(P(string))
    return exp(-1.0/len(str) * self.get_prob(str,unknown_substituted))

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
  return ''.join(choice(chars) for x in range(size))
    
if __name__ == "__main__":
  #mod = NGramModel(3,'lap')
  # mod = NGramModel(1,'lap','first')
  # corpus = [ [ 1, 2, 3, 1, 2, 4, 2, 4 ] ]
  # mod.train(corpus)
  # print mod.vocab_size()
  # print mod.freq
  # print mod.vocab_dict()
  # print mod.get_rand_word( [3] )
  # print exp(mod.get_cond_prob( [2,3] ))
  # print exp(mod.get_prob( [2,3] ))
  #print exp(mod.get_prob( [2,3]*20000 ))
  #print mod.laplacian_smoothing( [5], 3 )
  
  # print "Speed test:"
  # sizes = [1000]
  # for s in sizes:
    # words = dict( [ (i,id_generator()) for i in xrange(s/10) ] )
    # corpus = [ [ words[randint(0,s/10 - 1)] for i in xrange(s)] ]
    # mod = NGramModel(4,'gte','none','n')
    # profile.run("mod.train(corpus)")
    # profile.run("mod.get_perplexity( corpus[0] )")
    
  corpus = ["hello i am a dog".split(" "), "i am a cat".split(" ")]
  mod = NGramModel(2,'lap',None)
  mod.train(corpus)
  print exp(mod.get_prob("i am a pigeon".split(" ")))
  print (mod.get_perplexity("i am a pigeon".split(" ")))
  
  mod = NGramModel(2,'lap',None)
  mod.train([corpus[0]])
  mod.train([corpus[1]])
  print exp(mod.get_prob("i am a pigeon".split(" ")))
  print (mod.get_perplexity("i am a pigeon".split(" ")))
  
  


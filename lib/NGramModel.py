from random import random
from math import floor,log,exp


class NGramModel():
  def __init__(self, n, smooth_type=None):
    # self.freq:  dict( word_list => (#occurrances, dict(word=>frequency) ) )
    self.freq = dict({tuple(): (0,dict())})
    self.n = n
    if smooth_type == None:
      self.smooth = self.no_smoothing
    elif smooth_type == 'lap':
      self.smooth = self.laplacian_smoothing
    elif smooth_type == 'gte':
      self.smooth = self.good_turing_smoothing
    else:
      raise Exception("Invalid smoothing function name")
      
    
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
    # corpus: list of documents, each document is a list of words
    for doc in corpus:
      lst = []
      nm1 = self.n-1
      for w in doc:
        lst = [] if self.n == 1 else lst[-nm1:]
        lst.append(w)
        assert len(lst) <= self.n
        self.add_all_ntuple(lst)
      while len(lst) > 0: # Add trailing words
        lst = lst[1:]
        self.add_all_ntuple(lst)
  
  def get_rand_word( self, tup ):
    # Given n-1 words, get the n-th word
    head = tuple(tup[ -(self.n-1): ])
    if head in self.freq:
      (count,d) = self.freq[head]
      # TODO: Assuming no smoothing
      pos = random() * (count)
      for (w,f) in d.iteritems():
        if pos < f:
          return w
        else:
          pos -= f
      assert False #Never should get here!
      
      # TODO: Assuming Laplacian smoothing
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
    else:
      pos = int(floor(random() * (self.vocab_size())))
      v = self.vocab_dict().keys()
      assert pos < len(v)
      return v[pos]
  
  def vocab_size(self):
    empty = tuple()
    (count,d) = self.freq[empty]
    return len(d)
    
  def vocab_dict(self):
    empty = tuple()
    (count,d) = self.freq[empty]
    return d
  
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
    # returns P(tail | head)
    pass
    return 1
    
  def get_cond_prob( self, tup ):
    # Given n words, get log( P( wn | w1,...,wn-1 ) )
    # Given more than n words, only use the last n words and ignores the rest
    if len(tup) < 1:
      raise Exception("get_cond_prob: tup should not be an empty list")
    tup = tup[-self.n:]
    head = tup[:-1]
    tail = tup[-1]
    return log(self.smooth(head,tail))
  
  # P( w1 w2 ... wm ) = P(w1) P(w2 | w1) P(w3 | w1, w2) ... P(wn | w1,...,wn-1) P(wn+1 | w2,...,wn) ...
  def get_prob( self, str ):
    # Return log of probability
    return 1 if len(str) == 0 else self.get_prob( str[:-1] ) + self.get_cond_prob( str )

if __name__ == "__main__":
  mod = NGramModel(3,'lap')
  corpus = [ [ 1, 2, 3, 1, 2, 4, 2, 4 ] ]
  mod.train(corpus)
  print mod.vocab_size()
  print mod.freq
  print mod.vocab_dict()
  print mod.get_rand_word( [3] )
  print exp(mod.get_prob( [2,3] ))
  print mod.laplacian_smoothing( [5], 3 )
  


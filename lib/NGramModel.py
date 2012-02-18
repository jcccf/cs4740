
class NGramModel():
  def __init__(self, n, smooth_type=None)
    self.freq = dict()
    if smooth_type == None:
      self.smooth = lambda x : x
    elif smooth_type == 'lap':
      self.smooth = self.laplacian_smoothing
    elif smooth_type == 'gte':
      self.smooth = self.good_turing_smoothing
    else:
      raise Exception("Invalid smoothing function name")
    
  def add_ntuple(tup):
    head = tup[:-1]
    tail = tup[-1]
    if head in self.freq:
      d = (self.freq[ head ])
      if tail in d:
        d[ tail ] += 1
      else:
        d[ tail ] = 1
    else:
      self.freq[head] = dict( tail : 1 )

  def add_all_ntuple(tup):
    # tup = (w1, w2, w3, ..., wn)
    # adds (w1), (w1 w2), (w1, w2, w3) ...
    pass
  
  def get_rand_word( tup ):
    # Given n-1 words, get the n-th word
    pass
    
  def get_cond_prob( tup ):
    # Given n words, get P( wn | w1,...,wn-1 )
    # Given more than n words, only use the last n words and ignores the rest
    
    pass
  
  # P( w1 w2 ... wm ) = P(w1) P(w2 | w1) P(w3 | w1, w2) ... P(wn | w1,...,wn-1) P(wn+1 | w2,...,wn) ...
  def get_prob( str ):
    return get_prob( str[:-1] ) * get_cond_prob( str )
    
  

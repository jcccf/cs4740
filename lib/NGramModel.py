
class NGramModel():
  def __init__(self, n, smooth_fun)
    self.freq = dict()
    self.smooth_fun = smooth_fun
    
  def add_ntuple(tup):
    if tup[:-1] in self.freq:
      d = (self.freq[ tup[:-1] ])
      if tup[-1] in (self.freq[ tup[:-1] ]):
        d[ tup[-1] ] += 1
      else:
        d[ tup[-1] ] = 1
    else:
      self.freq[tup[:-1]] = dict( tup[-1] : 1 )

      
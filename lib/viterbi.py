import operator
from collections import Counter

#
# Another HMM Implementation
#

class Viterbi:
  def __init__(self, yy_p, xy_p):
    self.yy_p = yy_p # P(pos_i|pos_i-1,...) POS tag transition probabilities, a dictionary of dictionaries (pos_i -> (pos_i-1,... -> prob))
    self.xy_p = xy_p # P(word_i|pos_i,...) Probabilities of words given POS tags
    self.gs = len(self.yy_p.values()[0].keys()[0]) # Determine n in n-gram (=n-1) for transition probabilities
    self.em_gs = len(self.xy_p.values()[0].keys()[0]) # Determine n in n-gram (=n-1) for emission probabilities
    self.ykeys = [k for k in self.yy_p.values()[0].keys() if None not in k] # Get all non-None combinations of keys
    self.ytags = set() # Get all POS tags
    for tup in self.ykeys:
      for t in tup:
        self.ytags.add(t)
    self.ytags = list(self.ytags)
    
    if self.em_gs > self.gs:
      raise Exception("Emission Dependence cannot be greater than Transmission Dependence")
    print "Transition NGrams:", (self.gs+1), "\tEmission NGrams:", (self.em_gs+1)
    print "Keys:", self.ykeys
    print "Tags:", self.ytags
    
  def get_yy(self, tup):
    '''Return Transition Probability P(a|b,c,d...) of tup=(a,b,c,d...), extending tup if required'''
    if not isinstance(tup, tuple):
      tup = (tup,)
    if len(tup) - 1 < self.gs:
      tup += (None,) * (self.gs - (len(tup)-1))
    elif len(tup) - 1 > self.gs:
      tup = tup[:self.gs+1]
    return self.yy_p[tup[0]][tup[1:]]
    
  def get_xy(self, x, tup):
    '''Return Emission Probability P(x|a,b,c...) of x and tup=(a,b,c,...), extending tup if required'''
    if not isinstance(tup, tuple):
      tup = (tup,)
    if len(tup) < self.em_gs:
      tup += (None,) * (self.em_gs - len(tup))
    elif len(tup) > self.em_gs:
      tup = tup[:self.em_gs]
    return self.xy_p[x][tup]
    
  def get_T(self, T, i, tup):
    '''Return Computed Value of T[i][tup], extending tup'''
    if not isinstance(tup, tuple):
      tup = (tup,)
    if len(tup) < self.gs:
      tup += (None,) * (self.gs - len(tup))
    elif len(tup) > self.gs:
      tup = tup[:self.gs]
    return T[i-1][tup]
    
  def extend_gs(self, tup):
    '''Extend a tuple to the length self.gs, padding with None'''
    if not isinstance(tup, tuple):
      tup = (tup,)
    if len(tup) < self.gs:
      tup += (None,) * (self.gs - len(tup))
    elif len(tup) > self.gs:
      tup = tup[:self.gs]
    return tup

  def predict(self, xs):
    '''xs is an array of words'''
    # Initialize array
    T = [{a : 0.0 for a in self.ykeys} for j in range(len(xs))] # Set all values to 0.0
    T_prev = [{a : None for a in self.ykeys} for j in range(len(xs))] # Back-pointers
    
    # Build up table using DP
    i = 0
    for x in xs:
      print "i = ", i
      if i == 0: # Fill in the base
        for k in [k for k in self.yy_p.values()[0].keys() if k[0] is not None]:
          T[0][k] = self.get_yy(k[0]) * self.get_xy(x, k[0])
      elif i < self.gs: # Fill in the entries with some empty transitions
        # Store correct values for some certain tuples (MAY BE UNNECESSARY)
        for k2 in [k for k in self.yy_p.values()[0].keys() if (Counter(k)[None] == (self.gs-i))]:
          possibilities = []
          for k in self.ytags:
            tuply = k2[:i] + (k,)
            val = self.get_xy(x,tuply) * self.get_yy(tuply) * self.get_T(T, i, tuply[1:])
            possibilities.append((tuply[1:], val))
            print "\t\t", tuply, val
          T_prev[i][k2], T[i][k2] = max(possibilities, key = operator.itemgetter(1))
          print "\t", k2, "=>", T_prev[i][k2], T[i][k2]
        # Store correct values even for weird tuples that are longer than they are supposed to be
        for k2 in [k for k in self.yy_p.values()[0].keys() if (Counter(k)[None] < (self.gs-i))]:
          k2_lim = tuple([k if j < i + 1 else None for j,k in enumerate(k2)]) # Limit tuple so that wrong P_xy and P_yy values aren't read off
          T_prev[i][k2], T[i][k2] = self.extend_gs(k2[1:]), self.get_xy(x, k2_lim) * self.get_yy(k2_lim) * self.get_T(T, i, k2[1:])
          print "\t", k2, "=>", T_prev[i][k2], T[i][k2]
      else: # Fill in the rest of the entries with no empty transitions
        for k2 in self.ykeys:
          possibilities = []
          for k in self.ytags: # Loop through and pick the best
            tuply = k2 + (k,)
            val = self.get_xy(x, tuply) * self.get_yy(tuply) * self.get_T(T, i, tuply[1:])
            possibilities.append((tuply[1:], val))
            print "\t\t", tuply, val
          T_prev[i][k2], T[i][k2] = max(possibilities, key = operator.itemgetter(1))
          print "\t", k2, "=>", T_prev[i][k2], T[i][k2]
      i += 1
      
    # Generate prediction by following backpointers,
    # appending first element of each backpointer, then reversing at the end
    i -= 1
    generated = []
    result = max(T[i].iteritems(), key=operator.itemgetter(1))[0] # Get max of final probabilities
    generated.append(result[0])
    while i > 0:
      print result
      result = T_prev[i][self.extend_gs(result)]
      generated.append(result[0])
      i -= 1
    generated.reverse()
    return generated
    
    
if __name__ == '__main__':
  #
  # Examples Follow
  #
  
  # Transition 2-grams
  yy_p = { 
    "a": {(None,): 0.1, ("a",): 0.05, ("n",): 0.35, ("o",):0.1, ("t",):0.4},
    "n": {(None,): 0.4, ("a",): 0.1, ("n",): 0.05, ("o",):0.5, ("t",):0.1},
    "o": {(None,): 0.2, ("a",): 0.25, ("n",): 0.5, ("o",):0.1, ("t",):0.4},
    "t": {(None,): 0.3, ("a",): 0.6, ("n",): 0.1, ("o",):0.3, ("t",):0.1},
  }
  
  # Emission 2-grams
  xy_p = {
    "A": {("a",):0.4, ("n",):0.3, ("o",):0.1, ("t",):0.1},
    "T": {("a",):0.2, ("n",):0.1, ("o",):0.1, ("t",):0.4},
    "N": {("a",):0.1, ("n",):0.4, ("o",):0.1, ("t",):0.1},
    "Y": {("a",):0.2, ("n",):0.1, ("o",):0.2, ("t",):0.3},
    "W": {("a",):0.1, ("n",):0.1, ("o",):0.5, ("t",):0.1},
  }
  
  # ---
  
  # Transition 3-grams
  yy_p2 = { 
    "a": {(None,None): 0.8, ("a",None): 0.9, ("n",None): 0.4, ("a","a"): 0.1, ("a","n"): 0.1, ("n","a"): 0.2, ("n","n"): 0.5},
    "n": {(None,None): 0.2, ("a",None): 0.1, ("n",None): 0.6, ("a","a"): 0.9, ("a","n"): 0.9, ("n","a"): 0.8, ("n","n"): 0.5},
  }

  # Transition 4-grams
  yy_p3 = { 
    "a": {(None,None,None): 0.8, ("a",None,None): 0.9, ("n",None,None): 0.4, ("a","a",None): 0.1, ("a","n",None): 0.1, ("n","a",None): 0.2, ("n","n",None): 0.5, ("a","a","a"): 0.8, ("a","a","n"): 0.8, ("a","n","a"): 0.3, ("a","n","n"): 0.6, ("n","a","a"): 0.3, ("n","a","n"): 0.5, ("n","n","a"): 0.2, ("n","n","n"): 0.8},
    "n": {(None,None,None): 0.2, ("a",None,None): 0.1, ("n",None,None): 0.6, ("a","a",None): 0.9, ("a","n",None): 0.9, ("n","a",None): 0.8, ("n","n",None): 0.5, ("a","a","a"): 0.2, ("a","a","n"): 0.2, ("a","n","a"): 0.7, ("a","n","n"): 0.4, ("n","a","a"): 0.7, ("n","a","n"): 0.5, ("n","n","a"): 0.8, ("n","n","n"): 0.2},
  }

  # Emission 3-grams
  xy_p2 = {
    "A": {("a",None): 0.1, ("n",None): 0.1, ("a","a"):0.3, ("a","n"):0.2, ("n","a"):0.2, ("n","n"):0.1},
    "T": {("a",None): 0.1, ("n",None): 0.1, ("a","a"):0.1, ("a","n"):0.2, ("n","a"):0.3, ("n","n"):0.2},
    "N": {("a",None): 0.2, ("n",None): 0.2, ("a","a"):0.1, ("a","n"):0.1, ("n","a"):0.2, ("n","n"):0.2},
  }
  
  # Emission 2-grams
  xy_p1 = {
    "A": {(None,): 0.0, ("a",):0.1, ("n",):0.9},
    "T": {(None,): 0.0, ("a",):0.2, ("n",):0.8},
    "N": {(None,): 0.0, ("a",):0.1, ("n",):0.9},
  }
  
  # v = Viterbi(yy_p, xy_p)
  # print v.predict("TWY") # should give 'tot'
  
  # v = Viterbi(yy_p2, xy_p2)
  # print v.predict("AT") # should give aa (aa = 0.0072, na = 0.0016, an = 0.0024, nn = 0.0024 [reversed in output])
  # print v.predict("ATA") # should give aan (aaa = 2.16e-4, naa = 4.8e-5 [reversed in output])

  v = Viterbi(yy_p3, xy_p2)
  print v.predict("ATANT") # should give aannn (for N, aaa = 1.728e-5)
  
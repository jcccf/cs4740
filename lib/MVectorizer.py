from scipy.sparse import csr_matrix
from sklearn.preprocessing import Normalizer
from collections import Counter

class DictsVectorizer(object):
  '''Converts Dicts into a CSR Sparse Matrix'''
  '''Example: [{'hi':1, 'bye':2}, {'cool':2, 'bye':4}] is transformed into'''
  '''1 2 0'''
  '''0 4 2'''
  
  def __init__(self):
    self.k_to_i = {}
    self.matrix = None
  
  def dicts_to_csr(self, list_of_dicts):
    '''Converts a list of dictionaries to a CSR sparse matrix, with unique dictionary keys occupying unique columns'''
    counter = 0
    k_to_i = {}
    row = []
    col = []
    data = []
    for i, dicty in enumerate(list_of_dicts):
      for k, v in dicty.iteritems():
        data.append(v) # for the value itself
        row.append(i) # for the row number
        # and for the col number
        if k in k_to_i:
          col.append(k_to_i[k])
        else:
          k_to_i[k] = counter
          col.append(counter)
          counter += 1
    self.k_to_i = k_to_i
    self.matrix = csr_matrix((data,(row,col))) 
    return self.matrix
  
  def dicts_to_csr_transform(self, list_of_dicts):
    row = []
    col = []
    data = []
    for i, dicty in enumerate(list_of_dicts):
      for k, v in dicty.iteritems():
        if k in self.k_to_i:
          data.append(v) # for the value itself
          row.append(i) # for the row number
          col.append(self.k_to_i[k])
    return csr_matrix((data,(row,col)), shape=(len(list_of_dicts),self.matrix.shape[1]))
  
  def fit_transform(self, dicts):
    return self.dicts_to_csr(dicts)
    
  def transform(self, dicts):
    if self.matrix is None:
      raise Exception("Nothing fitted yet!")
    return self.dicts_to_csr_transform(dicts)

class ListsVectorizer(DictsVectorizer):
  '''Converts Lists into a CSR Sparse Matrix'''
  '''Example: [['hi', 'bye', 'bye'], ['cool', 'bye']] is transformed into'''
  '''1 2 0'''
  '''0 1 1'''

  def lists_to_csr(self, lists):
    '''Converts lists to a CSR sparse matrix, 1 row per list, with unique list elements occupying unique columns'''
    # convert list of lists into dictionary
    return self.dicts_to_csr([Counter(listy) for listy in lists])
  
  def lists_to_csr_transform(self, lists):
    return self.dicts_to_csr_transform([Counter(listy) for listy in lists])
    
  def fit_transform(self, lists):
    return self.lists_to_csr(lists)
    
  def transform(self, lists):
    if self.matrix is None:
      raise Exception("Nothing fitted yet!")
    return self.lists_to_csr_transform(lists)
  
#
# Random useful functions
#

def normalize(matrix):
  '''Normalize each row (L2-norm) of a CSR sparse matrix (it should work with most sparse matrices though)'''
  sparsy = matrix.tocoo()
  data = [float(d) for d in sparsy.data]
  return Normalizer().transform(csr_matrix((data, (sparsy.row, sparsy.col))))


#
# Simple tests
#

if __name__ == '__main__':
  a = {"hello": 1, "bye": 2}
  b = {"goodbye": 3, "hello": 4, "morning": 5}
  c = {"night": 6}
  d = {"hello": 7, "goodbye": 8}
  vec = DictsVectorizer()
  print vec.fit_transform([a,b,c,d]).todense()
  
  ax = [["bye", "hello", "bye"], ["goodbye", "hello", "morning"], ["night"], ["hello", "night", "night", "goodbye"]]
  ved = ListsVectorizer()
  print ved.fit_transform(ax).todense()
  print normalize(ved.fit_transform(ax)).todense()
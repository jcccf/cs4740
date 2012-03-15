from scipy.sparse import csr_matrix

class MVectorizer:
  def __init__(self, list_of_dicts):
    self.list_of_dicts = list_of_dicts
    
  def tocsr(self):
    counter = 0
    k_to_i = {}
    row = []
    col = []
    data = []
    for i, dicty in enumerate(self.list_of_dicts):
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
    return csr_matrix((data,(row,col)))

if __name__ == '__main__':
  a = {"hello": 1, "bye": 2}
  b = {"goodbye": 3, "hello": 4, "morning": 5}
  c = {"night": 6}
  d = {"hello": 7, "goodbye": 8}
  x = MVectorizer([a,b,c,d])
  print x.tocsr().todense()
from NGramModel import NGramModel

class RandomSentence():
  def __init__(self, model):
    self.model = model
  
  def gen_sentence(self,length):
    # Generates a sentence of length n
    # uses model.get_rand_word
    lst = []
    for k in xrange(0,length):
      lst.append(self.model.get_rand_word(lst))
    return lst
    
if __name__ == "__main__":
  mod = NGramModel(2)
  corpus = [ [ 1, 2, 3, 1, 2, 4, 2, 4 ] ]
  mod.train(corpus)
  
  rs = RandomSentence(mod)
  print rs.gen_sentence(10)

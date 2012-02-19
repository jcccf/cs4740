from NGramModel import NGramModel

class AuthorPrediction():
  def __init__(self, ngram, smooth_type=None):
    # ngram: number of grams to use
    self.author_model = dict()
    self.ngram = ngram
    self.smooth_type = smooth_type
  
  def add_author(self, name, sentences):
    self.author_model[name] = NGramModel(self.ngram, self.smooth_type)
    self.author_model[name].train(sentences)
  
  def predict_author(self, sentence):
    # Given the text, predict which author wrote it
    result = dict()
    for name, mod in self.author_model.iteritems():
      print "Computing for %s..." % name
      result[name] = mod.get_prob(sentence)
    return max(result, key=result.get)
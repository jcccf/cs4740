
class AuthorPrediction():
  def __init__(self, ngram, smooth_type=None):
    # ngram: number of grams to use
    self.author_model = dict()
    self.ngram = ngram
    self.smooth_type = smooth_type
  
  def add_author(name):
    self.author_model[name] = NGramModel(self.ngram, self.smooth_type)
    raise NotImplementedError("Justin needs to do this")
  
  def predict_author( str ):
    # Given the text, predict which author wrote it
    result = dict()
    for (name,mod) in self.author_model.iteritems():
      result[ name ] = mod.get_prob(str)
    return max(result, key=result.get)

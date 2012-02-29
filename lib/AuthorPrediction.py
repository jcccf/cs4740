from NGramModel import NGramModel

class AuthorPrediction():
  def __init__(self, ngram, smooth_type=None, unknown_type=None):
    # ngram: number of grams to use
    self.author_model = dict()
    self.ngram = ngram
    self.smooth_type = smooth_type
    self.unknown_type = unknown_type
  
  def add_author(self, name, sentences):
    self.author_model[name] = NGramModel(self.ngram, self.smooth_type, self.unknown_type)
    self.author_model[name].train(sentences)
  
  def predict_author(self, sentence, actual_label=None):
    # Given the text, predict which author wrote it
    result = {}
    if len(sentence) == 0: # If the sentence is blank, predict some arbitrary author
      # print "Blank sentence received"
      for i, (name, _) in enumerate(self.author_model.iteritems()):
        result[name] = i+1
    else:
      for name, mod in self.author_model.iteritems():
        if self.unknown_type == None or self.unknown_type == "none":
          mod.expand_vocab(sentence)
        result[name] = mod.get_perplexity(sentence)
        mod.set_vocab_expansion(0)
    if actual_label:
      rank = 1.0/(sorted(result, key=result.get).index(actual_label) + 1)
    else:
      rank = float("-inf")
    return (result, min(result, key=result.get), rank)
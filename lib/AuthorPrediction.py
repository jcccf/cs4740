from NGramModel import NGramModel
from WordParser import WordParser

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

if __name__ == '__main__':
  ap = AuthorPrediction(2, smooth_type='lap')
  co = WordParser('data/EnronDataset/train.txt', 'authors')
  for author, sentences in co.sentences().iteritems():
    ap.add_author(author, sentences)
  print "Training is done"
  co_test = WordParser('data/EnronDataset/test.txt', 'authors')
  co_test_sentences = co_test.sentences()
  print co_test_sentences.keys()
  print "Loaded Test Data"
  import itertools
  print ap.predict_author(list(itertools.chain.from_iterable(co_test_sentences['beck-s\t'])))
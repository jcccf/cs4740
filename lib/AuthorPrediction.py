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
  
  def predict_author(self, sentences):
    # Given the text, predict which author wrote it
    result = dict()
    for (name,mod) in self.author_model.iteritems():
      result[ name ] = mod.get_prob(sentences[0])
    return max(result, key=result.get)

if __name__ == '__main__':
  ap = AuthorPrediction(2)
  co = WordParser('data/EnronDataset/train.txt', 'authors')
  for author, sentences in co.sentences().iteritems():
    ap.add_author(author, sentences)
  co_test = WordParser('data/EnronDataset/test.txt', 'authors')
  co_test_sentences = co_test.sentences()
  print co_test_sentences.keys()
  ap.predict_author(co_test_sentences['beck-s\t'])
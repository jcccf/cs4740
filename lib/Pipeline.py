import nltk, Loader
from PipelineHelpers import *
from PipelineQuestions import *
from PipelineDocument import *

# The class to rule them all
class Answerer:
  
  def __init__(self, question_features, qno):
    self.qf = question_features.features(qno)
    self.qno = qno
    self.df = DocFeatures(qno)
    
  def answer(self):
    answers = []
    indices = self.df.filter_sentences(self.qf, doc_limit=20)
    for doc_index, sentence_index in indices:
      answers += self.df.match_nes(self.qf, doc_index, sentence_index)
      answers += self.df.match_wordnet(self.qf, doc_index, sentence_index)
    return answers
    
if __name__ == '__main__':
  qf = QuestionFeatures()
  a = Answerer(qf, 201)
  print a.answer()
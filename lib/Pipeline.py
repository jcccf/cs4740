import nltk, Loader
from PipelineHelpers import *
from PipelineQuestions import *
from PipelineDocument import *

# The class to rule them all
class Answerer:
  
  def __init__(self, question_features, qno):
    self.qf = question_features.features(qno)
    # print self.qf
    self.qno = qno
    self.df = DocFeatures(qno)
    
  def answer(self):
    answers = self.df.filter_sentences(self.qf, doc_limit=20)
    return answers
    
  def chunk(self,answers,chunksize=10,n_chunks=5):
    answers = answers[0: n_chunks*chunksize]
    n = 0
    chunks = []
    while len(answers) > 0:
      head = " ".join(answers[0:chunksize])
      answers = answers[chunksize:]
      chunks.append( head )
      n += 1
    while n < n_chunks:
      chunks.append( "nil" )
      n += 1
    return chunks
    
if __name__ == '__main__':
  for qno in range(201,400):
    qf = QuestionFeatures()
    a = Answerer(qf, qno)
    answers = a.answer()
    chunks = a.chunk(answers, n_chunks=5)
    print "\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )
    # print 
  
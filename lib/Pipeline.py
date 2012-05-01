import nltk, Loader, re, cProfile as profile
from PipelineHelpers import *
from PipelineQuestions import *
from PipelineDocument import *
from pprint import pprint

##
# untokenize: Joins a list of tokens back into a string
#
# Tokens are joined such that spaces are added between word tokens,
# but not between other tokens.
#
# This is to preserve tokens such as punctuation which are not
# pronounced but still needed for pauses in speech synthesis.
#
# @param tokens The list of tokens to join
##
def untokenize(tokens):
  acc = ''
  striplist = ["``", "'s", "''"]
  pattern = re.compile('\w+')
  for token in tokens:
    if pattern.match(token):
      acc = acc + ' ' + token
    elif token not in striplist:
      acc = acc + token
  acc = acc.replace('$ ',' $')
  return acc.strip()

# The class to rule them all
class Answerer:
  
  def __init__(self, question_features, qno):
    self.qf = question_features.features(qno)
    # print self.qf
    self.qno = qno
    self.df = DocFeatures(qno)
    self.stoplist = set( [("'s",), (".",), ("``","''"), ("'",)] )
    
  def answer(self):
    answers = self.df.filter_sentences(self.qf, doc_limit=20)
    # answers = self.df.filter_by_ne_corefs(self.qf, doc_limit=20)
    answers = [ a for a in answers if tuple(a) not in self.stoplist ]
    return answers
    
  def chunk(self,answers,chunksize=10,n_chunks=5):
    answers = answers[0: n_chunks*chunksize]
    n = 0
    chunks = []
    for i in range(n_chunks):
      chunk = []
      answers_copy = list(answers)
      for ans in answers_copy:
        untokenized_ans = untokenize(ans).split()
        if len(untokenized_ans) + len(chunk) <= chunksize:
          chunk.extend( untokenized_ans )
          answers.remove(ans)
          if len(chunk) == chunksize:
            break
      if len(chunk) == 0:
        chunk.append("nil")
      chunks.append( " ".join(chunk) )
      
    # while len(answers) > 0:
      # head = " ".join(answers[0:chunksize])
      # answers = answers[chunksize:]
      # chunks.append( head )
      # n += 1
    # while n < n_chunks:
      # chunks.append( "nil" )
      # n += 1
    return chunks
    
if __name__ == '__main__':
  for qno in range(201,399):
  # for qno in range(228,229):
    qf = QuestionFeatures()
    a = Answerer(qf, qno)
    # profile.run("Answerer(QuestionFeatures(), %d).answer()"%qno)
    answers = a.answer()
    # pprint(answers)
    chunks = a.chunk(answers, n_chunks=5)
    print "\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )
    # print 
  
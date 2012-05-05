import nltk, Loader, re, cProfile as profile
from PipelineHelpers import *
from PipelineQuestions import *
from PipelineDocument import *
from pprint import pprint
from WordNetDefinition import get_def_for_question_subject, lemmatize, lemmatizer
import argparse

# Set this to true to show debug output
PIPE_DEBUG = False

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
  
  def __init__(self, question, question_features, qno):
    self.question = question
    self.qf = question_features.features(qno)
    if PIPE_DEBUG: print "Question Features\n\t", self.qf
    self.qno = qno
    self.df = DocFeatures(qno)
    self.stoplist = set( [("'s",), (".",), ("``","''"), ("'",)] )
    
  def answer(self):
    wn_keywords = get_def_for_question_subject(self.question['question'], output="keywords")
    if PIPE_DEBUG: print "Wordnet Keywords\n\t", wn_keywords
    if wn_keywords != None:
      self.qf['keywords'] = lemmatize(remove_duplicates_list( self.qf['keywords'] + wn_keywords ))
      if PIPE_DEBUG: print "After Lemmatization\n\t", self.qf['keywords']
    
    answers = self.df.filter_sentences(self.qf, doc_limit=50)
    answers = [ a for a in answers if tuple(a) not in self.stoplist ]
    return answers
  
  def nonchunk(self,answers,n_chunks=5):
    # Doesn't chunk answers together.
    chunks = []
    for i in range(n_chunks):
      if i < len(answers):
        ans = answers[i]
        untokenized_ans = untokenize(ans).split()
        chunks.append( " ".join(untokenized_ans[:10]) )
      else:
        chunks.append("nil")
    return chunks
    
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
  argparser = argparse.ArgumentParser()
  # argparser.add_argument('-c', action='store_true', dest="chunk", help="chunk answers?")
  argparser.add_argument('-n', type=int, action='store', default=5, dest="n_chunks", help="no. of answers to give")
  argparser.add_argument('-l', type=int, action='store', default=400, dest="l", help="first question # to answer")
  argparser.add_argument('-u', type=int, action='store', default=600, dest="u", help="1 + last question # to answer")
  argparser.add_argument('-p', type=str, action='store', default="output_", dest="out_prefix", help="Prefix of output files")
  
  args = argparser.parse_args()
  
  qf = QuestionFeatures()
  questions = Loader.questions()
  f_nochunk = open(args.out_prefix+"nochunk.txt", 'w')
  f_chunk = open(args.out_prefix+"chunk.txt", 'w')
  for qno in range(args.l,args.u):
    a = Answerer(questions[qno], qf, qno)
    answers = a.answer()
    # pprint(answers)
    # if args.chunk:
      # chunks = a.chunk(answers, n_chunks=args.n_chunks)
    # else:
      # chunks = a.nonchunk(answers, n_chunks=args.n_chunks)
    # print "\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )
    
    chunks = a.nonchunk(answers, n_chunks=args.n_chunks)
    f_nochunk.write("\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )+"\n")
    # Only prints chunked version to stdout
    chunks = a.chunk(answers, n_chunks=args.n_chunks)
    f_chunk.write("\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )+"\n")
    print "\n".join( ["%d top_docs.%d "%(qno,qno) + chunk for chunk in chunks] )
    # print 
  # lemmatizer.save()
  f_nochunk.close()
  f_chunk.close()

# Loader - convenience functions to load data
import cPickle as pickle, nltk

#DIR = 'data/train'
DIR = 'data/test'

# Get questions
def questions():
  return pickle.load(open(DIR+'/parsed_questions.txt', 'rb'))

# Get CoreNLP-parsed questions  
def questions_core():
  return pickle.load(open(DIR+'/parsed_questions_core.txt', 'rb'))
  
# Get answers
def answers():
  return pickle.load(open(DIR+'/parsed_answers.txt', 'rb'))

# Get answers
def real_answers():
  return pickle.load(open(DIR+'/parsed_real_answers.txt', 'rb'))

# Get parsed docs
def docs(qno):
  return pickle.load(open(DIR+'/parsed_docs/top_docs.%d' % qno, 'rb'))

# Get CoreNLP-parsed docs
def docs_core(qno):
  return pickle.load(open(DIR+'/parsed_docs_core/top_docs.%d' % qno, 'rb'))

#
# Legacy Functions Below
#

# Try not to use this - here for legacy purposes
def docs_posne(qno):
  return pickle.load(open(DIR+'/parsed_docs_posne/top_docs.%d' % qno, 'rb'))

# Try not to use this either
def docs_tree(qno):
  return pickle.load(open(DIR+'/parsed_docs_tree/top_docs.%d' % qno, 'rb'))

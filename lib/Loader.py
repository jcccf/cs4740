# Loader - convenience functions to load data
import cPickle as pickle, nltk

# Get questions
def questions():
  return pickle.load(open('data/train/parsed_questions.txt', 'rb'))
  
# Get answers
def answers():
  return pickle.load(open('data/train/parsed_answers.txt', 'rb'))

# Get parsed docs with no other metadata
def docs(qno):
  return pickle.load(open('data/train/parsed_docs/top_docs.%d' % qno, 'rb'))

# Try not to use this - here for legacy purposes
def docs_posne(qno):
  return pickle.load(open('data/train/parsed_docs_posne/top_docs.%d' % qno, 'rb'))

# Use this
def docs_tree(qno):
  return pickle.load(open('data/train/parsed_docs_tree/top_docs.%d' % qno, 'rb'))

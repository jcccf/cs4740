import nltk, Loader
from PipelineHelpers import *
from CoreNLPLoader import *

# Load documents for only one specific question
class DocFeatures:
  def __init__(self, qno):
    self.docs = CoreNLPLoader(qno)

  # Return a set of indices of candidate sentences based on question features
  # Limit search to the top doc_limit docs
  # Return a list of (doc_index, sentence_index) tuples
  def filter_sentences(self, question_features, doc_limit=20):
    # TODO
    # Maybe, match keywords to NEs and Doc Corefs
    return []
    
  # Given a candidate sentence and question features, pick out
  # NEs that satisfy the question category
  def match_nes(self, question_features, doc_index, sentence_index):
    # TODO
    return []
    
  # Given a candidate sentence and question features, use WordNet
  # to pick out NPs that satisfy the question category
  def match_wordnet(self, question_features, doc_index, sentence_index):
    # TODO
    return []
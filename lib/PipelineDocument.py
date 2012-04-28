import nltk, Loader, itertools
from PipelineHelpers import *
from CoreNLPLoader import *
from collections import defaultdict
from QuestionClassifier import liroth_to_corenlp
from pprint import pprint

# Load documents for only one specific question
class DocFeatures:
  def __init__(self, qno):
    self.docs = CoreNLPLoader(qno)

  # Return a set of indices of candidate sentences based on question features
  # Limit search to the top doc_limit docs
  # Return a list of (tokenized sentence)
  def filter_sentences(self, question_features, doc_limit=20):
    # if filter_type == "answer_type":
      # return self.filter_by_answer_type(question_features, doc_limit)
    # elif filter_type == "keywords":
      # return self.filter_by_keyword_count(question_features, doc_limit)
    # else:
      # raise NotImplementedException()
    indices = self.filter_by_keyword_count(question_features, doc_limit)
    # pprint(indices)
    indices = self.filter_by_answer_type(question_features, indices)
    return indices
    
  def filter_by_keyword_count(self, question_features, doc_limit=20):
    # TODO
    # Maybe, match keywords to NEs and Doc Corefs
    keywords = question_features["keywords"]
    global_matches = []
    
    for doc_idx in range(0, min(doc_limit,len(self.docs.docs)) ):
      # Loop through each document
      paragraphs = self.docs.load_paras(doc_idx)
      # paragraphs = list of CoreNLPFeatures
      for paragraph_idx,paragraph in enumerate(paragraphs):
        sentences = paragraph.tokenized()
        # Loop through paragraphs
        matches = DocFeatures.naive_filter_sentences(keywords, sentences)
        matches = [ (count,doc_idx,paragraph_idx,sent_idx) for sent_idx,count in matches ]
        global_matches.extend(matches)
    
    # sort the matches by counts
    global_matches = sorted( global_matches, key=lambda x: -x[0] )
    global_matches = [ (doc_idx,paragraph_idx,sent_idx) for (count,doc_idx,paragraph_idx,sent_idx) in global_matches ]
    return global_matches
  
  def filter_by_answer_type(self, question_features, indices):
    question_classification = question_features['classification']
    answer_type = liroth_to_corenlp(question_classification)
    global_matches = []
    
    for doc_idx,paragraph_idx,sent_idx in indices:
      paragraphs = self.docs.load_paras(doc_idx)
      paragraph = paragraphs[paragraph_idx]
      named_entities = paragraph.named_entities()
      nes_in_sentence = named_entities[sent_idx]
      for words,ne_type in nes_in_sentence:
        if ne_type == answer_type or answer_type == None:
          global_matches.append( words )
    global_matches = [tuple(x) for x in global_matches]
    set_matches = set(global_matches)
    filtered_matches = []
    for w in global_matches:
      if w in set_matches:
        set_matches.remove(w)
        filtered_matches.append(w)
    return filtered_matches
  
  # Naive sentence filtering - looks for keywords in a sentence,
  # and returns a list of (index, # of keywords present) tuples
  @staticmethod
  def naive_filter_sentences(keywords, sentences):
    matches = []
    keywords = [keyword.lower() for keyword in keywords]
    for i, sentence in enumerate(sentences):
      # Generate a word frequency hash for this sentence
      word_hash = defaultdict(int)
      for word in sentence:
        word_hash[word.lower()] += 1
      # Count the number of appearances of keywords in this sentence
      count = 0
      for keyword in keywords:
        if keyword in word_hash:
          count += word_hash[keyword]
      if count > 0:
        matches.append((i, count))
    return matches
    
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
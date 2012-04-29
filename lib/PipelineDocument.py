import nltk, Loader, itertools
from PipelineHelpers import *
from CoreNLPLoader import *
from collections import defaultdict
from QuestionClassifier import liroth_to_corenlp,liroth_to_wordnet
from pprint import pprint
from nltk.corpus import wordnet as wn

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
    indices1 = self.filter_by_keyword_count(question_features, doc_limit)
    indices2 = self.filter_by_ne_corefs(question_features, doc_limit)
    indices = DocFeatures.union_sort(indices1, indices2)
    # pprint(indices)
    indices = self.filter_by_answer_type(question_features, indices)
    #indices = self.filter_by_wordnet(question_features, indices)
    return indices
  
  @staticmethod
  def union_sort(i1, i2):
    i = list(i1)
    i1hash = dict( [ ((x,y,z),True) for c,x,y,z in i1 ] )
    for c,x,y,z in i2: # Add stuff from i2 if it doesn't appear in i1
      if (x,y,z) not in i1hash:
        i.append((c,x,y,z))
    i = sorted(i, key = lambda x: -x[0]) # sort by count descending
    i = [ (x,y,z) for w,x,y,z in i ] # get rid of counts
    return i
  
  # TODO can match more exactly (ex. match only "The Golden Gate Bridge" vs "Directors of the Golden Gate Bridge District")
  # Filters by matching NEs in question to words in coreference clusters in paragraphs,
  # returning all sentences belonging to each matched cluster
  # also returns keyword counts for each sentence 
  def filter_by_ne_corefs(self, question_features, doc_limit=20):
    nes = question_features['nes']
    keywords = question_features["keywords"]
    global_matches = []
    # Loop through each document
    for doc_idx in range(0, min(doc_limit,len(self.docs.docs))):
      paragraphs = self.docs.load_paras(doc_idx)
      for para_idx, paragraph in enumerate(paragraphs):
        # Match clusters in this Paragraph
        clus_matches, sentence_indices = [], []
        clusters = paragraph.coreferences()
        sentences = paragraph.tokenized()
        if clusters is not None:
          for clus_idx, cluster in enumerate(clusters):
            for cluster_pair in cluster:
              for stringy, sentence_index, x, y, z in cluster_pair:
                # Match this cluster if for any string in this cluster,
                # that all words in any NE are present
                ne_match = True
                for ne_words, _ in nes:
                  for ne_word in ne_words:
                    if ne_word not in stringy:
                      ne_match = False
                if ne_match is True:
                  clus_matches.append(clus_idx)
          clus_matches = set(clus_matches)
          # Add sentence indices for each matched cluster
          for clus_idx in clus_matches:
            cluster = clusters[clus_idx]
            for cluster_pair in cluster:
              for _, sentence_index, x, y, z in cluster_pair:
                sentence_indices.append(sentence_index)
          sentence_indices = set(sentence_indices)
          for sentence_index in sentence_indices:
            # Get keyword count, add 1 to bias slightly
            count = DocFeatures.naive_filter_sentences(keywords, [sentences[sentence_index]], filter_zero=False)[0][1] + 1
            global_matches.append((count, doc_idx, para_idx, sentence_index))
    return global_matches
  
  # Returns sentences that contain keywords from the question, ordered by
  # the number of times keywords appear in a question
  def filter_by_keyword_count(self, question_features, doc_limit=20):
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
  def naive_filter_sentences(keywords, sentences, filter_zero=True):
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
      if count > 0 or filter_zero is False:
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

  # goes up the hypernym relation until either an element in answer_types is
  # a hypernym of the sense and returns true, or there are no more hypernyms
  # in which case it returns false
  def wordnet_hypernym_recursion(self, sense, answer_types):
    if sense in answer_types:
      return True
    else:
      hypernyms = sense.hypernyms()
      for hypernym in hypernyms:
        found = self.wordnet_hypernym_recursion(hypernym, answer_types)
        if found:
          return True
      return False

  def filter_by_wordnet(self, question_features, indices):
    question_classification = question_features['classification']
    coarse_question_class = question_classification.split(':')[0]
    answer_types = liroth_to_wordnet(question_classification)
    global_matches = []
    for doc_idx,paragraph_idx,sent_idx in indices:
      paragraphs = self.docs.load_paras(doc_idx)
      paragraph = paragraphs[paragraph_idx]
      tokenized_sentences = paragraph.tokenized()
      tokens = tokenized_sentences[sent_idx]
      for token in tokens:
        if answer_types == None:
          global_matches.append([token])
        else:
          token_synsets = wn.synsets(token)
          found = False
          for sense in token_synsets:
            if coarse_question_class in ['ENTY','NUM']:
              # this part is for words that are not names, like 'craters', 'dinosaurs'
              # might be applicable for definition and entity type questions
              found = self.wordnet_hypernym_recursion(sense, answer_types)
              if found:
                break
            else:
              # this part is for words that are names, like 'Australia'
              # names require the use of instance_hypernym() to get what they are
              instance_synsets = sense.instance_hypernyms()
              for instance_sense in instance_synsets:
                found = self.wordnet_hypernym_recursion(instance_sense, answer_types)
                if found:
                  break
          if found:
            global_matches.append([token]) 
    global_matches = [tuple(x) for x in global_matches]
    set_matches = set(global_matches)
    filtered_matches = []
    for w in global_matches:
      if w in set_matches:
        set_matches.remove(w)
        filtered_matches.append(w)
    return filtered_matches
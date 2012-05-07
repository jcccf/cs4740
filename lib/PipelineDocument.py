import nltk, Loader, itertools
from PipelineHelpers import *
from CoreNLPLoader import *
from Pipeline import *
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
    words = []
    
    # Get sentence indices, filtering by both keywords and NEs + corefs
    indices1 = self.filter_by_keyword_count(question_features, doc_limit)
    if PIPE_DEBUG: print "Indices from Keyword Count\n\t", indices1
    indices2 = self.filter_by_ne_corefs(question_features, doc_limit)
    if PIPE_DEBUG: print "Indices from NE Corefs\n\t", indices2
    indices3 = self.filter_by_exact_np_matches(question_features, doc_limit)
    if PIPE_DEBUG: print "Indices from Exact NP Matches\n\t", indices3
    indices = DocFeatures.union_sort(indices1, indices2)
    indices = DocFeatures.union_sort(indices3, indices)
    indices = [ (x,y,z) for w,x,y,z in indices ]
    if PIPE_DEBUG: print "Indices Combined\n\t", indices
    if PIPE_DEBUG:
      print "Index/Sentence Map"
      for index in indices:
        doc_idx,paragraph_idx,sent_idx = index
        paragraphs = self.docs.load_paras(doc_idx)
        paragraph = paragraphs[paragraph_idx]
        print index, "=>", paragraph.sentences()[sent_idx]
    
    # Attempt to find answer types using NEs and WordNet
    # Order NE answer types before WordNet results
    # But if this is definitely a description question, this is not going to help, so ignore
    if not self.is_description(question_features):
      if PIPE_DEBUG: print "Not a naive description question"
      words = self.filter_by_answer_type(question_features, indices)
      if PIPE_DEBUG: print "Words by answer type\n\t", words
      # words2 = self.filter_by_wordnet(question_features, indices) # Doesn't seem to work well :(
      # if PIPE_DEBUG: print "Words by wordnet type\n\t", words2
      # words = DocFeatures.match_prioritize(words, words2)
      # words = DocFeatures.union_order(words, words2)
      # pprint(words)
    
    # Pad results with NPs from sentences
    # words.extend(self.filter_by_nps(question_features, indices)) # Just extract NPs
    words.extend(self.filter_by_nps_nearby(question_features, indices)) # Extract in order of NPs near NEs
    if PIPE_DEBUG: print "Words with nearby NPs\n\t", words

    return words
  
  # Union the lists i1 and i2, sorting by the first index of each list element, descending
  @staticmethod
  def union_sort(i1, i2):
    i = list(i1)
    i1hash = dict( [ ((x,y,z),True) for c,x,y,z in i1 ] )
    for c,x,y,z in i2: # Add stuff from i2 if it doesn't appear in i1
      if (x,y,z) not in i1hash:
        i1hash[(x,y,z)] = True
        i.append((c,x,y,z))
    i = sorted(i, key = lambda x: -x[0]) # sort by count descending
    # i = [ (x,y,z) for w,x,y,z in i ] # get rid of counts
    return i
  
  # Union the lists i1 and i2, ensuring that all elements in i1 come before those in i2
  @staticmethod
  def union_order(i1, i2):
    i = list(i1)
    ihash = dict([(x,True) for x in i1])
    for y in i2:
      if y not in ihash:
        ihash[y] = True
        i.append(y)
    return i
  
  # Return a reordered list of i1, where we prioritize elements of i1 that also appear in i2
  @staticmethod
  def match_prioritize(i1, i2):
    highs, lows = [], []
    ihash = dict([(x,True) for x in i2])
    for x in i1:
      if x in ihash:
        highs.append(x)
      else:
        lows.append(x)
    return highs + lows
      
  # Tries to identify some very specific description questions
  def is_description(self, question_features):
    pos = question_features['pos']
    # WP is/was NN/P ? (ex. Who was Quetzacoatl?)
    if len(pos) == 4:
      if pos[0][1] == "WP" and (pos[1][0] == "is" or pos[1][0] == "was") and "NN" in pos[2][1]:
        return True
    return False
  
  # Filters by exact NP matches of NPs in the question to NPs in a sentence. Weighs these results more heavily,
  # since these are essentially phrase matches rather than individual word matches
  def filter_by_exact_np_matches(self, question_features, doc_limit=20):
    global_matches = []
    parse_tree = question_features['parse_tree']
    nps = extract_nps_without_determiners(parse_tree)
    phrases = [[w[0] for w in np] for np in nps]
    phrase_regexes = []
    for phrase in phrases:
      # Generate Regex for this phrase
      pre = re.compile("".join([w+"[\s]+" for w in phrase])[:-5])
      phrase_regexes.append((pre, 2**(2*(len(phrase)-1))))
    for doc_idx in range(0, min(doc_limit,len(self.docs.docs))):
      paragraphs = self.docs.load_paras(doc_idx)
      for para_idx, paragraph in enumerate(paragraphs):
        sentences = paragraph.sentences()
        tokenized_sentences = paragraph.tokenized()
        matches = naive_filter_sentences_phrases(phrase_regexes, sentences, tokenized_sentences)
        matches = [ (count,doc_idx,para_idx,sent_idx) for sent_idx,count in matches ]
        global_matches.extend(matches)        
    return global_matches
  
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
          # Sanity check since CoreNLP might mess up coref sentence indexing
          if len(sentence_indices) > 0 and max(sentence_indices) < len(sentences):
            for sentence_index in sentence_indices:
              # Get keyword count, add 1 to bias slightly
              try:
                count = naive_filter_sentences(keywords, [sentences[sentence_index]], filter_zero=False)[0][1] + 1
                global_matches.append((count, doc_idx, para_idx, sentence_index))
              except:
                print "DOCIDX"
                print doc_idx
                print "PARAINDEX"
                print para_idx
                print "PARA SENTENCES"
                print paragraph.sentences()
                print "KEYWORDS"
                print keywords
                print sentence_index
                print len(sentences)
                print "SENTENCES"
                print sentences
                print "CLUSTERS"
                print clusters
                print sentences[sentence_index]
                print naive_filter_sentences(keywords, [sentences[sentence_index]], filter_zero=False)
                raise Exception()
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
        # sentences = paragraph.tokenized()
        sentences = paragraph.lemmas()
        # Loop through paragraphs
        matches = naive_filter_sentences(keywords, sentences)
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
        if ne_type == answer_type: # or answer_type == None:
          global_matches.append( words )
    global_matches = [tuple(x) for x in global_matches]
    set_matches = set()
    filtered_matches = []
    for w in global_matches:
      if w not in set_matches:
        set_matches.add(w)
        filtered_matches.append(w)
    return filtered_matches
  
  # Return NPs of sentences, given sentence indices, filtering out NEs that appear
  # in the question itself
  def filter_by_nps(self, question_features, indices):
    word_filter = list(itertools.chain.from_iterable([w for w,t in question_features['nes']]))
    global_matches = []
    for doc_idx,paragraph_idx,sent_idx in indices:
      paragraphs = self.docs.load_paras(doc_idx)
      paragraph = paragraphs[paragraph_idx]
      sentence_parse_tree = paragraph.parse_trees(flatten=True)[sent_idx]
      nps_in_sentence = naive_extract_nps(sentence_parse_tree, word_filter)
      nps_in_sentence = [ [w for w,p in np] for np in nps_in_sentence]
      global_matches.extend(nps_in_sentence)
    return global_matches
  
  # Return NPs of sentences, given sentence indices, filtering out NEs that appear
  # in the question itself, and prioritze NPs that are near NPs containing the question's NEs
  # (+/-1 NP away)
  def filter_by_nps_nearby(self, question_features, indices):
    word_filter = list(itertools.chain.from_iterable([w for w,t in question_features['nes']]))
    high_matches, low_matches = [], []
    for doc_idx,paragraph_idx,sent_idx in indices:
      paragraphs = self.docs.load_paras(doc_idx)
      paragraph = paragraphs[paragraph_idx]
      sentence_parse_tree = paragraph.parse_trees(flatten=True)[sent_idx]
      nps_in_sentence = naive_extract_nps(sentence_parse_tree)
      nps_in_sentence = [ tuple([w for w,p in np]) for np in nps_in_sentence]
      # Pick NPs near NEs and prioritize them
      high, low = [], []
      for i, np in enumerate(nps_in_sentence):
        for words, word_type in question_features['nes']:
          matched = True
          for word in words:
            if word not in np: matched = False
          if matched is True:
            if i+1 < len(nps_in_sentence) and nps_in_sentence[i+1] not in high:
              high.append(nps_in_sentence[i+1])
            if nps_in_sentence[i] not in high:
              high.append(nps_in_sentence[i])
            if i-1 >= 0 and nps_in_sentence[i-1] not in high:
              high.append(nps_in_sentence[i-1])
      for np in nps_in_sentence:
        if np not in high: high.append(np)
      high_matches.extend(high)
      low_matches.extend(low)
      high_matches = [[w for w in np if w not in word_filter] for np in high_matches]
      low_matches = [[w for w in np if w not in word_filter] for np in low_matches]
    return high_matches + low_matches
    
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

# Helper functions for Pipeline
import nltk, itertools, re
from collections import defaultdict

stopwords = set(nltk.corpus.stopwords.words('english'))
# with open('data/stopwords.txt', 'r') as f:
#   for l in f:
#     stopwords.add(l.strip())

# Extract NPs from a flattened parse tree, removing stopwords
def extract_nps(parse_tree):
  # print parse_tree
  nps = []
  for elt in parse_tree:
    try:
      if elt.node == "NP":
        nps.append(elt.leaves())
    except:
      pass
  nps_filtered = []
  nps = [[w for w in np if w[0].lower() not in stopwords] for np in nps]
  nps = [np for np in nps if len(np) > 0]
  return nps

# Extract NPs without determiners from a parse tree.
# For example, "the longest river" becomes "longest river"
def extract_nps_without_determiners(parse_tree):
  nps = []
  for elt in parse_tree:
    try:
      if elt.node == "NP":
        nps.append(elt.leaves())
    except:
      pass
  new_nps = []
  for np in nps:
    while len(np) > 0 and np[0][1] == 'DT':
      np.pop(0)
    if len(np) > 0:
      new_nps.append(np)
  return new_nps
  
# Return a list of keywords from a flattened parse tree
def extract_keywords(parse_tree):
  nps = extract_nps(parse_tree)
  return [w[0] for w in list(itertools.chain.from_iterable(nps))]

# Extract NPs from sentence, removing stopwords and words in the word filter
def naive_extract_nps(sentence_ptree, word_filter = []):
  word_filter = [word.lower() for word in word_filter]
  nps = []
  for elt in sentence_ptree:
    try:
      if elt.node == "NP":
        nps.append(elt.leaves())
    except:
      pass
  nps_filtered = []
  nps = [[w for w in np if w[0].lower() not in stopwords and w[0].lower() not in word_filter ] for np in nps]
  nps = [np for np in nps if len(np) > 0]
  return nps

# Naive sentence filtering - looks for keywords in a sentence,
# and returns a list of (index, # of keywords present) tuples
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
      # matches.append((i, float(count) / len(sentence)))
      matches.append((i, count))
  return matches
  
# Naive sentence filtering - look for phrases in an UNTOKENIZED sentence
# and returns a list of (index, # of keywords present (special squared value)) tuples
def naive_filter_sentences_phrases(phrase_regexes, sentences, tokenized_sentences):
  matches = []
  # phrase_regexes = []
  # for phrase in phrases:
    ## Generate Regex for this phrase
    # pre = re.compile("".join([w+"[\s]+" for w in phrase])[:-5])
    # phrase_regexes.append((pre, 2**(2*(len(phrase)-1))))
  for i, sentence in enumerate(sentences):
    count = 0
    for pre, prelen in phrase_regexes:
      if pre.search(sentence):
        count += prelen
    if count > 0:
      # matches.append((i, float(count) / len(tokenized_sentences[i])))
      matches.append((i, count))
  return matches
    
def naive_filter_sentences_unweighted(keywords, sentences, filter_zero=True):
  matches = []
  keywords = [keyword.lower() for keyword in keywords]
  for i, sentence in enumerate(sentences):
    word_hash = dict( [(word.lower(),True) for word in sentence] )
    count = 0
    for keyword in keywords:
      if keyword in word_hash: count += 1
    if count > 0 or filter_zero is False: matches.append((i, count))
  return matches
  
def remove_duplicates_list(l):
  s = set()
  acc = []
  for w in l:
    if w not in s:
      s.add(w)
      acc.append(w)
  return acc

# DocFilterer - a rudimentary IR system to find relevant sentences
from collections import defaultdict
import QuestionParser, itertools

# Naive sentence filtering - looks for keywords in a sentence,
# and returns a list of (index, # of keywords present) tuples
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

def naive_filter_sentences_unweighted(keywords, sentences):
  matches = []
  keywords = [keyword.lower() for keyword in keywords]
  for i, sentence in enumerate(sentences):
    word_hash = { word:True for word in sentence }
    count = 0
    for keyword in keywords:
      if keyword in word_hash: count += 1
    if count > 0: matches.append((i, count))
  return matches

# Extract NPs from sentence, removing stopwords and the keywords themselves
def naive_extract_nps(keywords, sentence_ptree):
  keywords = [keyword.lower() for keyword in keywords]
  nps = []
  for elt in sentence_ptree:
    try:
      if elt.node == "NP":
        nps.append(elt.leaves())
    except:
      pass
  nps_filtered = []
  nps = [[w for w in np if w[0].lower() not in QuestionParser.stopwords and w[0].lower() not in keywords] for np in nps]
  nps = [np for np in nps if len(np) > 0]
  return nps

# Convert NPs to 10-word chunks
def nps_to_chunks(nps, word_size=10):
  chunks = []
  words = [[n[0] for n in np] for np in nps]
  words = list(itertools.chain.from_iterable(words))
  chunk = []
  for word in words:
    if len(chunk) < word_size:
      chunk.append(word)
    else:
      chunks.append(" ".join(chunk))
      chunk = [word]
  if len(chunk) > 0:
    chunks.append(" ".join(chunk))
  return chunks
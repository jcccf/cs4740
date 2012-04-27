# Helper functions for Pipeline
import nltk, itertools

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
  
# Return a list of keywords from a flattened parse tree
def extract_keywords(parse_tree):
  nps = extract_nps(parse_tree)
  return [w[0] for w in list(itertools.chain.from_iterable(nps))]

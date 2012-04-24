# -*- coding: utf-8 -*-
# Question Parser - extracts useful information about a question
import cPickle as pickle, nltk, itertools

stopwords = set(nltk.corpus.stopwords.words('english'))
# with open('data/stopwords.txt', 'r') as f:
#   for l in f:
#     stopwords.add(l.strip())

# Extract NPs from a question, removing stopwords
def extract_nps(question_tree):
  # print question_tree
  nps = []
  for elt in question_tree:
    try:
      if elt.node == "NP":
        nps.append(elt.leaves())
    except:
      pass
  nps_filtered = []
  nps = [[w for w in np if w[0].lower() not in stopwords] for np in nps]
  nps = [np for np in nps if len(np) > 0]
  return nps
  
# Return a list of keywords
def extract_keywords(question_tree):
  nps = extract_nps(question_tree)
  return [w[0] for w in list(itertools.chain.from_iterable(nps))]
  
# TODO Could use this to rank keywords in order of importance
def extract_keywords_ranked(question_tree):
  brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news', simplify_tags=True)
  word_fd = nltk.FreqDist(word for (word, tag) in brown_news_tagged)
  raise NotImplementedError

if __name__ == "__main__":
  q = pickle.load(open('data/train/parsed_questions.txt', 'rb'))
  print extract_keywords(q[201]['parse_tree'])

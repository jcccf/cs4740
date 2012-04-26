# The Real Parse Tree
# As we can observe from this small example, ParseTrees take way too long to train.
# http://nltk.googlecode.com/svn/trunk/doc/howto/parse.html
import nltk
from nltk.corpus import treebank

print "Loading Productions"
productions = []
for fileid in treebank.fileids()[:100]:
  for t in treebank.parsed_sents(fileid):
    # print t.productions()
    productions += t.productions()

print "Inducing PCFG..."
grammar = nltk.induce_pcfg(nltk.nonterminals('S')[0], productions)

print "Generating..."
a = "The company is large."
w = nltk.word_tokenize(a)
p = nltk.pos_tag(w)
for t in nltk.parse.pchart.InsideChartParser(grammar).nbest_parse(w):
  print t
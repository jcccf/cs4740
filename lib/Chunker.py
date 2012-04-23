# -*- coding: utf-8 -*-
# Chunker to generate parse trees - do not import unless you really need it!

# To get parse trees, call Chunker.chunker.parse(ps),
# where ps is a pos-tagged sentence

import nltk.chunk, itertools
from nltk.tag import UnigramTagger, BigramTagger
from nltk.corpus import treebank_chunk

def backoff_tagger(train_sents, tagger_classes, backoff=None):
  for cl in tagger_classes:
     backoff = cl(train_sents, backoff=backoff)
  return backoff

def conll_tag_chunks(chunk_sents):
  tagged_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
  return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

class TagChunker(nltk.chunk.ChunkParserI):
  def __init__(self, train_chunks, tagger_classes=[UnigramTagger, BigramTagger]):
    train_sents = conll_tag_chunks(train_chunks)
    self.tagger = backoff_tagger(train_sents, tagger_classes)
    
  def parse(self, tagged_sent):
    if not tagged_sent: return None
    (words, tags) = zip(*tagged_sent)
    chunks = self.tagger.tag(tags)
    wtc = itertools.izip(words, chunks)
    return nltk.chunk.conlltags2tree([(w,t,c) for (w,(t,c)) in wtc])

# To see how good this is
def evaluate_chunker():
  train_chunks = treebank_chunk.chunked_sents()[:3000]
  test_chunks = treebank_chunk.chunked_sents()[3000:]
  chunker = TagChunker(train_chunks)
  score = chunker.evaluate(test_chunks)
  print score.accuracy()
  
# Initialize chunker
train_chunks = treebank_chunk.chunked_sents()
chunker = TagChunker(train_chunks)
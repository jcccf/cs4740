import WordParser, NGramModel, RandomSentence
from math import exp

cor = WordParser.WordParser('data/fbis/fbis.train')
print "done"
mod = NGramModel.NGramModel(5)
mod.train([cor.words()])
# print mod.vocab_size()
# print mod.vocab_dict()
# print exp(mod.get_cond_prob(['I', 'am', 'as']))

ran = RandomSentence.RandomSentence(mod)
for i in range(10):
  print ran.gen_sentence(10)
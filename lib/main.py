import WordParser, NGramModel, RandomSentence
from math import exp

cor = WordParser.WordParser('data/Dataset3/Train.txt')
print "done"
mod = NGramModel.NGramModel(3)
mod.train([cor.words()])
# print mod.vocab_size()
# print mod.vocab_dict()
# print exp(mod.get_cond_prob(['I', 'am', 'as']))

ran = RandomSentence.RandomSentence(mod)
print ran.gen_sentence(10)
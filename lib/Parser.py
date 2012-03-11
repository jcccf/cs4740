import re, StringIO, nltk
from lxml import etree
from lxml import html

class Example:
  def __init__(self, word, pos, senses, context_before, target, context_after):
    self.word = word
    self.pos = pos
    self.senses = senses
    self.context_before = context_before
    self.cb_tokenized = None
    self.target = target
    self.context_after = context_after
    self.ca_tokenized = None
    
  def valid_senses(self, dicty):
    '''Return the valid senses of this particular word based on the senses binary list'''
    return [d for i, d in enumerate(dicty['%s.%s' % (self.word, self.pos)]) if self.senses[i] == 1]
    
  def all_senses(self, dicty):
    '''Return all senses of this particular word'''
    return dicty['%s.%s' % (self.word, self.pos)]
    
  def __load_tokenized(self):
    if self.cb_tokenized is None:
      self.cb_tokenized = nltk.word_tokenize(self.context_before)
      self.cb_tokenized.reverse()
      self.ca_tokenized = nltk.word_tokenize(self.context_after)     
  
  def word_positions(self, word):
    '''Give a list of all positions of a word in context relative to the target word (negative means it came before, positive means came after)'''
    self.__load_tokenized()
    pos = []
    pos += [-(i+1) for i, x in enumerate(self.cb_tokenized) if x == word]
    pos += [i+1 for i, x in enumerate(self.ca_tokenized) if x == word]
    return pos
    
  def count_within_window(self, word, n=None):
    '''Return the number of times a word appears within N words of the target word'''
    if n is not None:
      x = [1 for i in self.word_positions(word) if abs(i) <= n]
    else:
      x = self.word_positions(word)
    return len(x)
    
  def words_positions(self):
    '''Give a list of (position,word) tuples of all words in context'''
    self.__load_tokenized()
    wp = []
    for i, w in enumerate(self.cb_tokenized):
      wp.append((-(i+1), w))
    for i, w in enumerate(self.cb_tokenized):
      wp.append(((i+1), w))
    return sorted(wp)
    
def load_examples(filename='data/wsd-data/train.data'):
  line_matcher = re.compile(r'([\w]+)\.([\w]+) ([0-9 ]+)@[ ]*(.*)@([\w]+)@(.*)')
  examples = []
  with open(filename, 'r') as f:
    for l in f:
      match_obj = line_matcher.match(l)
      if match_obj:
        word, pos, senses, context_before, target, context_after = match_obj.groups()
        senses = [int(b) for b in senses.strip().split(' ')]
        examples.append(Example(word, pos, senses, context_before, target, context_after))
      else:
        raise Exception("Example Regex Failed to Match on\n%s" % l)
  return examples

class Sense:
  def __init__(self, synset, gloss):
    self.synset = synset
    self.gloss = gloss
  
  def __repr__(self):
    return "Sense(synset=%s, gloss=%s)" % (self.synset, self.gloss)

def load_dictionary(filename="data/dictionary-mapping.xml"):
  parser = etree.XMLParser()
  with open(filename, 'r') as f:
    myfile = f.read()
  
  pattern=re.compile("gloss=\"(.*)\"");
  for group in pattern.findall(myfile):
    myfile = myfile.replace(str(group), str(group).replace("\"", "'"))
  sio = StringIO.StringIO(myfile)
  
  tree = etree.parse(sio, parser)
  dictionary = {}
  for lexelt in tree.xpath("//lexelt"):
    dictionary[lexelt.get("item")] = [Sense(sense.get("synset"), sense.get("gloss")) for sense in lexelt]
  
  return dictionary

# Temporarily put here just to load the test data for use by some other code
# To be deleted later
def load_test_data(filename):
  line_matcher = re.compile(r'([\w]+)\.([\w]+) ([0-9 ]+)@[ ]*(.*)@([\w]+)@(.*)')
  examples = []
  with open(filename, 'r') as f:
    for l in f:
      match_obj = line_matcher.match(l)
      if match_obj:
        word, pos, senses, context_before, target, context_after = match_obj.groups()
        senses = [int(b) for b in senses.strip().split(' ')]
        examples.append(Example(word, pos, senses, context_before, target, context_after))
      else:
        raise Exception("Example Regex Failed to Match on\n%s" % l)
  return examples

# The examples are organized by words and part of speech in a dictionary
# this is so that each word and part of speech can be trained separately
def load_training_data(filename):
  line_matcher = re.compile(r'([\w]+)\.([\w]+) ([0-9 ]+)@[ ]*(.*)@([\w]+)@(.*)')
  dictexamples = dict()
  with open(filename, 'r') as f:
    for l in f:
      match_obj = line_matcher.match(l)
      if match_obj:
        word, pos, senses, context_before, target, context_after = match_obj.groups()
        senses = [int(b) for b in senses.strip().split(' ')]
        if (word,pos) in dictexamples:
            dictexamples[(word,pos)].append(Example(word, pos, senses, context_before, target, context_after))
        else:
            dictexamples[(word,pos)] = [Example(word, pos, senses, context_before, target, context_after)]
      else:
        raise Exception("Example Regex Failed to Match on\n%s" % l)
  return dictexamples

if __name__ == '__main__':
  egs = load_examples()
  dictionary = load_dictionary()  

  print egs[0].word
  print egs[0].pos
  print egs[0].senses
  print egs[0].context_before
  print egs[0].context_after
  print "---"
  print egs[0].valid_senses(dictionary)
  print egs[0].all_senses(dictionary)
  print "---"
  print egs[0].words_positions()
  print egs[0].word_positions('the')
  print egs[0].count_within_window('the', 5)
import re, StringIO, nltk, cPickle as pickle, hashlib, string, itertools
from nltk.corpus import wordnet as wn
from lxml import etree
from lxml import html
import os

# Make cache directories if they don't exist
try:
  os.makedirs('data/lesk/')
except:
  pass
try:
  os.makedirs('data/lesk_wordlist/')
except:
  pass
try:
  os.makedirs('data/pos/')
except:
  pass


wordnet_tags = {'NN':wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}
def penn_to_wn(penn_tag):
  if penn_tag[:2] in wordnet_tags:
    return wordnet_tags[penn_tag[:2]]
  else:
    return None

class WordSet:
  tokenizer = nltk.tokenize.RegexpTokenizer('[^\w\']+', gaps=True)

  def __init__(self, words):
    words = WordSet.tokenizer.tokenize(words)
    self.words = [w for w in words if w not in Example.stopwords]
    
  def overlap(self, other_sets):
    count = 0
    for other_set in other_sets:
      for word in other_set.words:
        if word in self.words:
          count += 1
    total_words = sum([len(other_set.words) for other_set in other_sets])
    return (count, total_words)

class Example:
  
  stopwords = nltk.corpus.stopwords.words('english')
  
  def __init__(self, word, pos, senses, context_before, target, context_after):
    self.word = word
    self.pos = pos
    self.senses = senses
    self.context_before = context_before
    self.cb_tokenized = None
    self.target = target
    self.context_after = context_after
    self.ca_tokenized = None
    self.lesk_vector = None
    self.lesk_wordlist = None
    self.cache()
  
  def hash(self):
    return hashlib.md5(self.target+self.context_before+self.context_after).hexdigest()
  
  def cache(self):
    filehash = hashlib.md5(self.target+self.context_before+self.context_after).hexdigest()
    try:
      s = pickle.load(open('data/pos/%s' % filehash, 'r'))
      self.cb_tokenized = s['cb_tokenized']
      self.ca_tokenized = s['ca_tokenized']
      self.posf = s['posf']
    except:
      print "Caching Tokenizations and POS for %s" % self.target
      self.__load_tokenized()
      self.__load_pos()
      s = {'cb_tokenized': self.cb_tokenized, 'ca_tokenized': self.ca_tokenized, 'posf': self.posf}
      pickle.dump(s, open('data/pos/%s' % filehash, 'w'))
      
  def lesk(self, dicty, window_size=100):
    '''Return overlaps of each sense with senses of surrounding words (window size of 2)
      Stopwords are ignored, and results are normalized to the maximum overlap observed
      Note that the size of the sense vector returned is 1 less, because index 0 was reserved for an unknown sense'''
    if self.lesk_vector is None:
      filehash = self.hash()
      try:
        self.lesk_vector = pickle.load(open('data/lesk/%d_%s' % (window_size, filehash), 'r'))
      except:
        print "Caching Lesk for window size %d" % window_size
        self.lesk_vector = self.__load_lesk_vector(dicty, window_size=window_size)
        pickle.dump(self.lesk_vector, open('data/lesk/%d_%s' % (window_size, filehash), 'w'))
    return self.lesk_vector
  
  def lesk_words(self, dicty, window_size=2):
    '''Return a list of all non-stopwords that appear in the definitions of words surrounding the target'''
    if self.lesk_wordlist is None:
      filehash = self.hash()
      try:
        self.lesk_wordlist = pickle.load(open('data/lesk_wordlist/%d_%s' % (window_size, filehash), 'r'))
      except:
        print "Caching Lesk words for window size %d" % window_size
        # Generate words in definitions of words surrounding target
        other_words = []
        words = self.words_window(window_size)
        for word, pos in words:
          # print word, pos
          baseword = wn.morphy(word)
          if baseword is not None:
            pos = penn_to_wn(pos)
            synsets = wn.synsets(baseword, pos=pos) if pos is not None else wn.synsets(baseword)
            for synset in synsets:
              other_words.append(WordSet(synset.definition).words)
        self.lesk_wordlist = list(itertools.chain.from_iterable(other_words))
        pickle.dump(self.lesk_wordlist, open('data/lesk_wordlist/%d_%s' % (window_size, filehash), 'w'))
    return self.lesk_wordlist

  def __load_lesk_vector(self, dicty, window_size=100):
    # print example.word, example.pos, example.target
    # print example.senses
  
    # Generate WordSets of surrounding words
    other_sets = []
    words = self.words_window(window_size)
    for word, pos in words:
      # print word, pos
      baseword = wn.morphy(word)
      if baseword is not None:
        pos = penn_to_wn(pos)
        synsets = wn.synsets(baseword, pos=pos) if pos is not None else wn.synsets(baseword)
        for synset in synsets:
          other_sets.append(WordSet(synset.definition))
  
    # for sety in other_sets:
    #   print sety.words
  
    # Loop through possible wordsets and note counts:
    counts = []
    for sense in self.all_senses(dicty):
      curr_set = WordSet(sense.gloss)
      # print curr_set.words
      counts.append(curr_set.overlap(other_sets))
    # print counts
  
    # Normalize and return
    countfirsts = [count[0] for count in counts]
    countfirsts_max = max(countfirsts)
    if countfirsts_max > 0:
      return [float(count)/countfirsts_max for count in countfirsts]
    else:
      return [0.0 for count in countfirsts]

  
  def valid_senses(self, dicty):
    '''Return the valid senses of this particular word based on the senses binary list'''
    return [d for i, d in enumerate(dicty['%s.%s' % (self.word, self.pos)]) if self.senses[i] == 1]
    
  def all_senses(self, dicty):
    '''Return all senses of this particular word'''
    return dicty['%s.%s' % (self.word, self.pos)]
    
  def __load_tokenized(self):
    if self.cb_tokenized is None:
      self.cb_tokenized = nltk.word_tokenize(self.context_before)
      # self.cb_tokenized.reverse()
      self.ca_tokenized = nltk.word_tokenize(self.context_after)     
  
  def word_positions(self, word):
    '''Give a list of all positions of a word in context relative to the target word (negative means it came before, positive means came after)'''
    pos = []
    cb_offset = len(self.cb_tokenized)
    pos += [-(cb_offset-i) for i, x in enumerate(self.cb_tokenized) if x == word]
    pos += [i+1 for i, x in enumerate(self.ca_tokenized) if x == word]
    return pos
    
  def words_window(self, size=2, remove_stopwords=True):
    words = []
    lencb = len(self.cb_tokenized)
    lenca = len(self.ca_tokenized)
    for i in range(size):
      if len(self.cb_tokenized) > i:
        word = self.cb_tokenized[-(i+1)].lower()
        if word not in string.punctuation and word not in Example.stopwords:
          words.append((word, self.posf[lencb-1-i][1]))
      if len(self.ca_tokenized) > i:
        word = self.ca_tokenized[i].lower()
        if word not in string.punctuation and word not in Example.stopwords:
          words.append((word, self.posf[lencb+i][1]))
    return words
  
  def __load_pos(self):
    self.__load_tokenized()
    text = self.cb_tokenized + [self.target] + self.ca_tokenized
    cb_len, ca_len = len(self.cb_tokenized), len(self.ca_tokenized)
    poses = nltk.pos_tag(text)
    posf = []
    for i in range(cb_len):
      posf.append((-(cb_len-i), poses[i][1]))
    offset = cb_len + 1
    for i in range(ca_len):
      posf.append((i+1, poses[offset+i][1]))
    self.posf = posf
  
  def pos_positions(self, filter_punctuation=True, window=None, sentence=True):
    posf = list(self.posf)
    if filter_punctuation:
      posf = [(a,b) for a,b in posf if b is not "."]
    if window:
      posf = [(a,b) for a,b in posf if abs(a) <= window]
    posf = [str(a)+","+b for a,b in posf]
    if sentence is True:
      posf = " ".join(posf)
    return posf
    
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
    cb_offset = len(self.cb_tokenized)
    for i, w in enumerate(self.cb_tokenized):
      wp.append((-(cb_offset-i), w))
    for i, w in enumerate(self.ca_tokenized):
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
  
  for eg in egs:
    # print eg.lesk(dictionary)
    print eg.lesk_words(dictionary)

  # print egs[0].word
  # print egs[0].pos
  # print egs[0].senses
  # print egs[0].context_before
  # print egs[0].context_after
  # print "---"
  # print egs[0].valid_senses(dictionary)
  # print egs[0].all_senses(dictionary)
  # print "---"
  # print egs[0].words_positions()
  # print egs[0].word_positions('the')
  # print egs[0].count_within_window('the', 5)
  # print egs[0].pos_positions()
  # print egs[0].pos_positions(window=2)
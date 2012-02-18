from HTMLParser import HTMLParser
from collections import Counter
import nltk, itertools, string, re
try:
  import cPickle as pickle
except:
  import pickle

class WordParser:
  def __init__(self, filename):
    self.filename = filename
    self.frequencies = None
    self.inv_frequencies = None
    self.word_list = None
    self.sentence_list = None
    self.words()
    
  # http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
  class MLStripper(HTMLParser):
    def __init__(self):
      self.reset()
      self.fed = []
    def handle_data(self, d):
      self.fed.append(d)
    def get_data(self):
      return ''.join(self.fed)    
  def __strip_tags(self, html):
    s = WordParser.MLStripper()
    s.feed(html)
    return s.get_data()
    
  def __load_string(self):
    print "Parsing file..."
    with open(self.filename, 'r') as f:
      self.string = f.read()
    if "<html>" in self.string: # Remove HTML tags
      self.string = self.__strip_tags(self.string)
    elif '<' in self.string: # Else remove generic tags
      tag_re = re.compile(r'<.*?>')
      self.string = tag_re.sub('', self.string)
    
  def words(self):
    # Return list of words in order
    if not self.word_list:
      try:
        with open(self.filename+".words", 'r') as f:
          print "Loading words from pickle..."
          self.word_list = pickle.load(f)
      except IOError as e:
        print "Haven't loaded this file before, please wait!"
        self.__load_string()
        print "Tokenizing..."
        sentences = nltk.sent_tokenize(self.string)
        self.sentence_list = [[w.lower() for w in nltk.word_tokenize(s) if w not in string.punctuation] for s in sentences]
        self.word_list = list(itertools.chain.from_iterable(self.sentence_list))
        print "Writing to pickle..."
        pickle.dump(self.word_list, open(self.filename+".words", 'w'))
        pickle.dump(self.sentence_list, open(self.filename+".sentences", 'w'))
    return self.word_list
    
  def sentences(self):
    # Return a list of list of words in order (each word list is a sentence)
    if not self.sentence_list: # Lazy loading again
      with open(self.filename+".sentences", 'r') as f:
        print "Loading sentences from pickle..."
        self.sentence_list = pickle.load(f)
    return self.sentence_list
  
  def __load_frequencies(self):
    if not self.frequencies:
      print "Lazily loading frequencies..."
      self.frequencies = Counter(self.word_list)
  
  def freq(self, w):
    # Returns the frequency of word w
    self.__load_frequencies()
    if w in self.frequencies:
      return self.frequencies[w]
    else:
      return 0
    
  def freqs(self):
    # Returns a dictionary of word frequencies
    self.__load_frequencies()
    return self.frequencies
  
  def __load_inv_frequencies(self):
    if not self.inv_frequencies:
      print "Lazily loading inverse frequencies"
      self.__load_frequencies()
      self.inv_frequencies = {}
      for k, v in self.frequencies.iteritems():
        self.inv_frequencies.setdefault(v, []).append(k)
      self.inv_frequencies_count = { k:len(v) for k, v in self.inv_frequencies.iteritems()}
  
  def inv_freq(self, n):
    # Returns a count of words that appear n times
    self.__load_inv_frequencies()
    if n in self.inv_frequencies_count:
      return self.inv_frequencies_count[n]
    else:
      return 0
    
  def inv_freq_words(self, n):
    # Returns a list of words that appear n times
    self.__load_inv_frequencies()
    return self.inv_frequencies[n]

if __name__ == '__main__':
  w = WordParser('data/fbis/fbis.train')
  print w.sentences()
  print w.inv_freq(1)
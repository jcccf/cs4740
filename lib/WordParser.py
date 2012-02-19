from HTMLParser import HTMLParser
from collections import Counter
import nltk, itertools, string, re
try:
  import cPickle as pickle
except:
  import pickle

class WordParser(object):
  def __init__(self, filename):
    self.filename = filename
    self.frequencies = None
    self.inv_frequencies = None
    self.word_list = None
    self.sentence_list = None
    self._load()
    
  # http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
  class MLStripper(HTMLParser):
    def __init__(self):
      self.reset()
      self.fed = []
    def handle_data(self, d):
      self.fed.append(d)
    def get_data(self):
      return ''.join(self.fed)    
  def _strip_tags(self, html):
    s = WordParser.MLStripper()
    s.feed(html)
    return s.get_data()
    
  def _sanitize_list(self, doclist):
    newdoclist = []
    for doc in doclist:
      if "<html>" in doc: # Remove HTML tags
        doc = self._strip_tags(doc)
      elif '<' in doc: # Else remove generic tags
        tag_re = re.compile(r'<.*?>')
        doc = tag_re.sub('', doc)
      if len(doc.strip()) > 0:
        newdoclist.append(doc)
    return newdoclist
    
  def _load_string(self):
    with open(self.filename, 'r') as f:
      self.string = f.read()

  def _load(self):
    try:
      f = open(self.filename+".words", 'r')
      f.close()
    except IOError as e:
      print "Haven't loaded this file before, please wait!"
      print "Parsing file..."
      self._load_string()
      print "Tokenizing..."
      sentences = nltk.sent_tokenize(self.string)
      sentences = self._sanitize_list(sentences)
      self.sentence_list = [[w.lower() for w in nltk.word_tokenize(s) if w not in string.punctuation] for s in sentences]
      self.word_list = list(itertools.chain.from_iterable(self.sentence_list))
      print "Writing to pickle..."
      pickle.dump(self.word_list, open(self.filename+".words", 'w'))
      pickle.dump(self.sentence_list, open(self.filename+".sentences", 'w'))

  def _load_words(self):
    if not self.word_list:
      print "Lazily loading words from pickle..."
      self.word_list = pickle.load(open(self.filename+".words", 'r'))

  def words(self):
    # Return list of words in order
    if not self.word_list:
      self._load_words()
    return self.word_list
    
  def _load_sentences(self):
    if not self.sentence_list: # Lazy loading again
      print "Loading sentences from pickle..."
      self.sentence_list = pickle.load(open(self.filename+".sentences", 'r'))

  def sentences(self):
    # Return a list of list of words in order (each word list is a sentence)
    self._load_sentences()
    return self.sentence_list

  def _load_frequencies(self):
    if not self.frequencies:
      print "Lazily loading frequencies..."
      self._load_words()
      self.frequencies = Counter(self.word_list)
  
  def freq(self, w):
    # Returns the frequency of word w
    self._load_frequencies()
    if w in self.frequencies:
      return self.frequencies[w]
    else:
      return 0
    
  def freqs(self):
    # Returns a dictionary of word frequencies
    self._load_frequencies()
    return self.frequencies
  
  def _load_inv_frequencies(self):
    if not self.inv_frequencies:
      print "Lazily loading inverse frequencies..."
      self._load_frequencies()
      self.inv_frequencies = {}
      for k, v in self.frequencies.iteritems():
        self.inv_frequencies.setdefault(v, []).append(k)
      self.inv_frequencies_count = { k:len(v) for k, v in self.inv_frequencies.iteritems()}
  
  def inv_freq(self, n):
    # Returns a count of words that appear n times
    self._load_inv_frequencies()
    if n in self.inv_frequencies_count:
      return self.inv_frequencies_count[n]
    else:
      return 0
    
  def inv_freq_words(self, n):
    # Returns a list of words that appear n times
    self._load_inv_frequencies()
    return self.inv_frequencies[n]

class DocWordParser(WordParser):
  def __init__(self, filename):
    super(DocWordParser, self).__init__(filename)
    self.document_list = None
  
  def _load(self):
    try:
      f = open(self.filename+".words", 'r')
      f.close()
    except IOError as e:
      print "Haven't loaded this file before, please wait!"
      print "Parsing file..."
      self._load_string()
      
      print "Tokenizing..."
      
      # Split into documents
      docs = self.string.split('</DOC>')
      docs = self._sanitize_list(docs)
      
      self.sentence_list = []
      self.word_list = []
      self.document_list = []
      
      for doc in docs:
        sentences = nltk.sent_tokenize(doc)
        sentences = self._sanitize_list(sentences)
        sentence_list = [[w.lower() for w in nltk.word_tokenize(s) if w not in string.punctuation] for s in sentences]
        word_list = list(itertools.chain.from_iterable(sentence_list))
        self.document_list.append(sentence_list)
        self.word_list.append(word_list)
      self.sentence_list = list(itertools.chain.from_iterable(self.document_list))
      self.word_list = list(itertools.chain.from_iterable(self.word_list))
      print "Writing to pickle..."
      pickle.dump(self.word_list, open(self.filename+".words", 'w'))
      pickle.dump(self.sentence_list, open(self.filename+".sentences", 'w'))
      pickle.dump(self.document_list, open(self.filename+".docs", 'w'))
      
  def _load_docs(self):
    if not self.document_list: # Lazy loading again
      print "Lazily loading documents from pickle..."
      self.document_list = pickle.load(open(self.filename+".docs", 'r'))

  def docs(self):
    # Return a list of list of words in order (each word list is a sentence)
    self._load_docs()
    return self.document_list
  
class EnronWordParser(WordParser):
  def __init__(self, filename):
    super(EnronWordParser, self).__init__(filename)
    self.authors = None

  def author_sentences(self):
    # Return an (author, sentences) dictionary
    self._load_sentences()
    self._load_authors()
    auths = {}
    for a,s in zip(self.authors, self.sentence_list):
      auths.setdefault(a,[]).append(s)
    return auths
    
  def _load_authors(self):
    if not self.authors:
      print "Lazily loading authors from pickle..."
      self.authors = pickle.load(open(self.filename+".authors", 'r'))

  def _load(self):
    try:
      f = open(self.filename+".words", 'r')
      f.close()
    except IOError as e:
      print "Haven't loaded this file before, please wait!"
      self._load_string()
      print "Tokenizing..."
        
      # Split by line since one email on one line
      lines = self.string.splitlines()
      self.authors, sentences = [], []
      for l in lines:
        auth, sent = l.split('\t', 1)
        self.authors.append(auth)
        sentences.append(sent)
      pickle.dump(self.authors, open(self.filename+".authors", 'w'))
      
      self.sentence_list = [[w.lower() for w in nltk.word_tokenize(s) if w not in string.punctuation] for s in sentences]
      self.word_list = list(itertools.chain.from_iterable(self.sentence_list))
      print "Writing to pickle..."
      pickle.dump(self.word_list, open(self.filename+".words", 'w'))
      pickle.dump(self.sentence_list, open(self.filename+".sentences", 'w'))

if __name__ == '__main__':
  # w = EnronWordParser('data/EnronDataset/train.txt')
  # for a, s in w.author_sentences().iteritems():
  #   print a
  #   print len(s)
  # print w.inv_freq(1)
  w = DocWordParser('data/wsj/wsj.train')
  print w.docs()
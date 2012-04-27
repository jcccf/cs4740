import CoreNLPParser, Loader, sys, os, cPickle as pickle
from unidecode import unidecode

# Loads and Caches Documents using CoreNLP
class CoreNLPLoader():
  def __init__(self, qno):
    try:
      os.makedirs('data/train/parsed_docs_core')
    except:
      pass
    self.qno = qno
    self.cache()
    
  def cache(self):
    # Test for file existence, if so, just load it in
    try:
      self.docs = Loader.docs_core(self.qno)
    except:    
      # If not, load docs
      parser = CoreNLPParser.CoreNLPParser()
      docs = Loader.docs(self.qno)
      parsed_docs = []
      for doc in docs:
        print "Parsing Docno", doc['docno'],
        sys.stdout.flush()
        parsed_doc = { 'docno': doc['docno'] }
        for k in ['leadpara', 'headline', 'text']:
          if doc[k] is None: continue
          jsons = []
          for paragraph in doc[k]:
            print ".",
            sys.stdout.flush()
            json = {}
            while "sentences" not in json:
              json = parser.parse(unidecode(paragraph))
            jsons.append(json)
          parsed_doc[k] = jsons
        parsed_docs.append(parsed_doc)
        print "done"
        sys.stdout.flush()
      with open('data/train/parsed_docs_core/topdocs.%d' % self.qno, 'wb') as f:
        pickle.dump(parsed_docs, f)
      self.docs = parsed_docs
    
  # Load the document at index doc_index,
  # returning a hash of lists of CoreNLPFeatures objects (corresponding to paragraphs)
  def load_doc(doc_index, flatten=True):
    special_doc = { 'docno': self.docs[doc_index]['docno'] }
    for k in ['leadpara', 'headline', 'text']:
      if k in self.docs[doc_index]:
        special_doc[k] = [CoreNLPParser.CoreNLPFeatures(v) for v in self.docs[doc_index][k]]
    return special_doc
  
  # Simply load a list of CoreNLPFeatures objects from the document at index doc_index
  def load_paras(doc_index):
    special_doc = []
    for k in ['headline', 'leadpara', 'text']:
      if k in self.docs[doc_index]:
        special_doc += [CoreNLPParser.CoreNLPFeatures(v) for v in self.docs[doc_index][k]]
    return special_doc

if __name__ == '__main__':
  # See CoreNLPFeatures for more functions to call
  cl = CoreNLPLoader(201)
  a = cl.load_paras(0)
  print a[0].sentences()
  print a[0].coreferences()
  
  # # Run the below!
  # for i in range(300, 400):
  #     cl = CoreNLPLoader(i)   
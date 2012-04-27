import CoreNLPParser, Loader, sys
from unidecode import unidecode

try:
  os.makedirs('data/train/parsed_docs_core')
except:
  pass

class CoreNLPLoader():
  def __init__(self, qno):
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
    
  def load_doc(doc_index):
    special_doc = {}
    for k in ['leadpara', 'headline', 'text']:
      if k in self.docs[doc_index]:
        special_doc[k] = [CoreNLPParser.CoreNLPFeatures(v) for v in self.docs[doc_index][k]]
    return special_doc

if __name__ == '__main__':
  cl = CoreNLPLoader(201)
  a = cl.load_doc(0)
  
  # # Run the below!
  # for i in range(300, 400):
  #     cl = CoreNLPLoader(i)   
import CoreNLPParser, Loader, sys
from unidecode import unidecode

try:
  os.makedirs('data/train/parsed_docs_core')
except:
  pass

class CoreNLPLoader():
  def __init__(self, qno):
    self.qno = qno
    
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
    
  def load_doc(docindex):
    pass

if __name__ == '__main__':
  cl = CoreNLPLoader(201)
  cl.cache()
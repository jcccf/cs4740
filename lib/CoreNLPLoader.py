import CoreNLPParser, Loader, sys, os, cPickle as pickle
from unidecode import unidecode
import jsonrpc
import argparse

# Loads and Caches Documents using CoreNLP
class CoreNLPLoader():
  def __init__(self, qno, host="127.0.0.1", port=8080):
    self.host,self.port = host,port
    try:
      os.makedirs('data/train/parsed_docs_core')
    except:
      pass
    self.qno = qno
    for attempt in range(10):
        try:
            self.cache()
        except jsonrpc.RPCTransportError as e:
            print "Attempt %d timed out.."%attempt
            continue
        break
    
  def cache(self):
    # Test for file existence, if so, just load it in
    try:
      self.docs = Loader.docs_core(self.qno)
    except:
      # If not, load docs
      parser = CoreNLPParser.CoreNLPParser(host=self.host,port=self.port)
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
      with open('data/train/parsed_docs_core/top_docs.%d' % self.qno, 'wb') as f:
        pickle.dump(parsed_docs, f)
      self.docs = parsed_docs
    
  # Load the document at index doc_index,
  # returning a hash of lists of CoreNLPFeatures objects (corresponding to paragraphs)
  def load_doc(self, doc_index, flatten=True):
    special_doc = { 'docno': self.docs[doc_index]['docno'] }
    for k in ['leadpara', 'headline', 'text']:
      if k in self.docs[doc_index]:
        special_doc[k] = [CoreNLPParser.CoreNLPFeatures(v) for v in self.docs[doc_index][k]]
    return special_doc
  
  # Simply load a list of CoreNLPFeatures objects from the document at index doc_index
  def load_paras(self, doc_index):
    special_doc = []
    for k in ['headline', 'leadpara', 'text']:
      if k in self.docs[doc_index]:
        special_doc += [CoreNLPParser.CoreNLPFeatures(v) for v in self.docs[doc_index][k]]
    return special_doc

if __name__ == '__main__':
  # See CoreNLPFeatures for more functions to call
  # cl = CoreNLPLoader(201)
  # a = cl.load_paras(0)
  # print a[2].sentences()
  # print a[2].coreferences()
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-l', type=int, default=201, action='store', dest="lb", help="start parsing from this question number")
  argparser.add_argument('-u', type=int, default=399, action='store', dest="ub", help="stop parsing after this question number")
  argparser.add_argument('--port', type=int, default=8080, action='store', dest="port", help="port of server")
  argparser.add_argument('--host', default="127.0.0.1", action='store', dest="host", help="host name of server")
  args = argparser.parse_args()
  
  if args.lb <= args.ub:
    step = 1
    args.ub += 1
  else:
    step = -1
    args.ub -= 1
  
  # Run the below!
  for i in range(args.lb, args.ub, step):
    try:
      cl = CoreNLPLoader(i,host=args.host, port=args.port)
    except:
      continue
      
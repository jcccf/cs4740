import CoreNLPParser, Loader, sys, os, cPickle as pickle
from unidecode import unidecode
import jsonrpc
import argparse

DIR = Loader.DIR

class CoreNLPQuestionLoader():
  def __init__(self, host="127.0.0.1", port=8080):
    self.host,self.port = host,port
    self.cache()
    
  def cache(self):
    try:
      self.questions = Loader.questions_core()
    except:
      print "Parsing Questions..."
      parser = CoreNLPParser.CoreNLPParser(host=self.host,port=self.port)
      qs = Loader.questions()
      parsed_qs = {}
      for qno, q in qs.iteritems():
        json = parser.parse(unidecode(q['question']))
        parsed_qs[qno] = json    
      with open(DIR+'/parsed_questions_core.txt', 'wb') as f:
        pickle.dump(parsed_qs, f)
        self.questions = parsed_qs
  
  def load_question(self, qno):
    return CoreNLPParser.CoreNLPFeatures(self.questions[qno])

# Loads and Caches Documents using CoreNLP
class CoreNLPLoader():
  def __init__(self, qno, host="127.0.0.1", port=8080):
    self.host,self.port = host,port
    try:
      os.makedirs(DIR+'/parsed_docs_core')
    except:
      pass
    self.qno = qno
    try:
        self.cache()
    except:
        pass
    
  def cache(self):
    # Test for file existence, if so, just load it in
    try:
      self.docs = Loader.docs_core(self.qno)
    except:
      # If not, load docs
      print "Loading Docs for", self.qno
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
            # Move the retry here so that we don't have to re-parse
            # a whole lot of docs
            for attempt in range(3):
                try:
                  json = parser.parse(unidecode(paragraph))
                  break # for
                except jsonrpc.RPCTransportError as e:
                  if attempt+1 == 3:
                    print
                    print "---"
                    print unidecode(paragraph)
                    print "---"
                    sys.stdout.flush()
                    raise Exception()
                  continue # for
                except Exception as e:
                  if attempt+1 == 3:
                    print
                    print "---"
                    print e
                    print "---"
                    raise e
                  continue # for
            jsons.append(json)
          parsed_doc[k] = jsons
        parsed_docs.append(parsed_doc)
        print "done"
        sys.stdout.flush()
      with open(DIR+'/parsed_docs_core/top_docs.%d' % self.qno, 'wb') as f:
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
  argparser.add_argument('-q', default=False, action='store_true', dest="questions", help="parse questions instead of docs")
  argparser.add_argument('-l', type=int, default=201, action='store', dest="lb", help="start parsing from this question number")
  argparser.add_argument('-u', type=int, default=399, action='store', dest="ub", help="stop parsing after this question number")
  argparser.add_argument('--port', type=int, default=8080, action='store', dest="port", help="port of server")
  argparser.add_argument('--host', default="127.0.0.1", action='store', dest="host", help="host name of server")
  args = argparser.parse_args()
  
  if args.questions is True:
    cl = CoreNLPQuestionLoader()
    print cl.load_question(201).parse_trees(flatten=True)
  else:
    if args.lb <= args.ub:
      step = 1
      args.ub += 1
    else:
      step = -1
      args.ub -= 1
    for i in range(args.lb, args.ub, step):
      try:
        cl = CoreNLPLoader(i,host=args.host, port=args.port)
      except:
        continue
      

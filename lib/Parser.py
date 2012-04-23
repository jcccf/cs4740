import nltk, glob, re, json, cPickle as pickle, os.path
from lxml import etree
from lxml import html

#
# Useful formats you need to know
# /parsed_docs
#     List of { "docno": docno, "title": title, "leadpara": leadpara, "text": text}
#     * Note that values may be None if the document did not have that particular field
# /parsed_docs_posne
#     List of { "title": { "pos": List of sentences, "ne": List of nltk NE trees }, "leadpara": { "pos": List of sentences, "ne": List of nltk NE trees },...
#     * Same indices as /parsed_docs

# Parse nice files for each document
# docs = list of {docno, title, leadpara, text}
def parse_docs():
  print "Parsing Docs"
  try:
    os.makedirs('data/train/parsed_docs')
  except:
    pass

  parser = etree.XMLParser(recover=True)
  for filename in glob.glob("data/train/docs/top_docs.*"):
    print filename
    docs = []
    with open(filename, 'r') as f:
      data = f.read()
      xmldocs = re.split('[\s]*Qid:[\s]*[0-9]+[\s]*Rank:[\s]*[0-9]+[\s]*Score:[\s]*[0-9\.]+[\s]*', data) # Split the documents
      assert len(xmldocs) == 51
      xmldocs.pop(0) # Remove the first element which is blank
      for xmldoc in xmldocs:
        xmldoc = re.sub(r'<([a-zA-Z]+)[\s]+[a-zA-Z0-9= ]+[\s]*>', r'<\1>', xmldocs[1]) # Fix some broken XML
        tree = etree.XML(xmldoc, parser)
        docno = tree.xpath("//DOCNO")[0].text
        leadpara = tree.xpath("//LEADPARA")[0].text if len(tree.xpath("//LEADPARA")) > 0 else None
        headline = tree.xpath("//HEADLINE")[0].text if len(tree.xpath("//HEADLINE")) > 0 else None
        text = re.sub(r'\[[^\[\]]*\]', '', tree.xpath("//TEXT")[0].text) # Remove square brackets
        docs.append({ "docno": docno, "leadpara": leadpara, "text": text })
    # Write to a pickle
    with open('data/train/parsed_docs/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(docs, f)

# Generate part-of-speech tags and identify named entities
def generate_pos_ne():
  print "Generating POS Tags/NEs"
  try:
    os.makedirs('data/train/parsed_docs_posne')
  except:
    pass
  for filename in glob.glob("data/train/parsed_docs/top_docs.*"):
    print filename
    pdocs = []
    with open(filename, 'rb') as f:
      docs = pickle.load(f)
      for doc in docs:
        pdoc = {}
        for k in doc.keys(): # Generate POS tags and NEs for each key in the document
          if doc[k] is not None:
            # Split into sentences, then words, POS tag words, then chunk tagged words
            sentences = nltk.sent_tokenize(doc[k])
            allpos, allne = [], []
            for s in sentences:
              words = nltk.word_tokenize(s)
              postags = nltk.pos_tag(words)
              nes = nltk.ne_chunk(postags)
              allpos.append(postags)
              allne.append(nes)
            pdoc[k] = { "pos": allpos, "ne": allne }
          else:
            pdoc[k] = None
        pdocs.append(pdoc)
    # Write to a pickle
    with open('data/train/parsed_docs_posne/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(pdocs, f)

if __name__ == '__main__':
  # parse_docs()
  generate_pos_ne()
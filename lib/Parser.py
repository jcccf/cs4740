# -*- coding: utf-8 -*-
# Parser to parse documents, questions

# Useful formats you need to know
# /parsed_docs
#     List of { "docno": docno, "title": title, "leadpara": leadpara, "text": text}
#     * Note that values may be None if the document did not have that particular field
# /parsed_docs_posne
#     List of { 
#       "title": 
#         { "sentences": List of list of words, "pos": List of sentences, "ne": List of nltk NE trees, "parse_tree": Parse Trees of sentences }, 
#       "leadpara": { "pos": List of sentences, "ne": List of nltk NE trees },...
#     * Same indices as /parsed_docs
# /parsed_questions.txt
#     Dict of Question Number : { 
#         "question": question text, 
#         "raw_pos": pos tags,
#         "raw_ne": NE tree,
#         "qwords": list of question words,
#         "nouns": list of nouns in the question,
#         "nes": list of NEs, and these are tuples of (NE_TYPE, word)
#       }

import nltk, glob, re, json, cPickle as pickle, os.path
from lxml import etree
from lxml import html
import Chunker

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
      for xmldoc in xmldocs:
        xmldoc = re.sub(r'<([a-zA-Z]+)[\s]+[a-zA-Z0-9= ]+[\s]*>', r'<\1>', xmldoc) # Fix some broken XML
        if len(xmldoc.strip()) == 0:
          continue
        tree = etree.XML(xmldoc, parser)
        docno = tree.xpath("//DOCNO")[0].text
        leadpara = tree.xpath("//LEADPARA")[0].text if len(tree.xpath("//LEADPARA")) > 0 else None
        headline = tree.xpath("//HEADLINE")[0].text if len(tree.xpath("//HEADLINE")) > 0 else None
        if len(tree.xpath("//TEXT")) > 0:
          text = re.sub(r'<[a-zA-Z\/][^>]*>', '', etree.tostring(tree.xpath("//TEXT")[0])) # Remove XML/HTML Tags
          text = re.sub(r'\[[^\[\]]*\]', '', text) # Remove square brackets
        else:
          text = None
        docs.append({ "docno": docno, "leadpara": leadpara, "text": text, "headline": headline })
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
          if doc[k] is not None and doc[k] is not "docno":
            # Split into sentences, then words, POS tag words, then chunk tagged words
            sentences = nltk.sent_tokenize(doc[k])
            allsent, allpos, allne, allptree = [], [], [], []
            for s in sentences:
              words = nltk.word_tokenize(s)
              postags = nltk.pos_tag(words)
              nes = nltk.ne_chunk(postags)
              tree = Chunker.chunker.parse(postags)
              allsent.append(words)
              allpos.append(postags)
              allne.append(nes)
              allptree.append(tree)
            pdoc[k] = { "sentences": allsent, "pos": allpos, "ne": allne, "parse_tree": allptree }
          else:
            pdoc[k] = None
        pdocs.append(pdoc)
    # Write to a pickle
    with open('data/train/parsed_docs_posne/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(pdocs, f)

# Identify the question type and parse out the NEs
def parse_questions():
  parsed_questions = {}
  with open('data/train/questions.txt', 'r') as f:
    data = f.read()
    questions = re.split('[\s]*</top>[\s]*', data)
    assert len(questions) == 200
    questions.pop()
    for question in questions:
      question_number = re.search(r"<num>[\s]*Number:[\s]*([0-9]+)", question).group(1)
      question = re.search(r"<desc>[\s]*Description:[\s]*([a-zA-Z0-9\-\?\'\. ]+)", question).group(1)
      question_words = nltk.word_tokenize(question)
      question_pos = nltk.pos_tag(question_words)
      question_nes = nltk.ne_chunk(question_pos)
      qwords, nouns, nes = [], [], []
      for part in question_nes:
        try:
          nes.append((part.node, part.leaves()[0][0]))
        except:
          if part[1] == 'WP' or part[1] == 'WRB':
            qwords.append(part[0])
          elif part[1] == 'NN' or part[1] == 'NNP':
            nouns.append(part[0])
      print qwords, nouns, nes
      print question_pos
      parsed_questions[question_number] = { "question": question, "raw_pos": question_pos, "raw_ne": question_nes, "qwords": qwords, "nouns": nouns, "nes": nes }
  with open('data/train/parsed_questions.txt', 'wb') as f:
    pickle.dump(parsed_questions, f)

if __name__ == '__main__':
  parse_docs()
  parse_questions()
  generate_pos_ne()
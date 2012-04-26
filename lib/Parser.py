# -*- coding: utf-8 -*-
# Parser to parse documents, questions

# Useful formats you need to know
# /parsed_docs
#   List of { "docno": docno, "title": title, "leadpara": leadpara, "text": text}
#   * Note that values may be None if the document did not have that particular field
# /parsed_questions.txt
#   Dict of Question Number : { 
#       "question": question text, 
#       "pos": pos tags,
#       "ne": NE tree,
#       "parse_tree": Parse tree of sentence
#       "question_words": list of question words,
#       "nouns": list of nouns in the question,
#       "ne_words": list of NEs, and these are tuples of (NE_TYPE, word)
#   }
# /parsed_docs_trees
#   List of { "text" : { 
#       "sentences": List of List of Words,
#       "pos_tags": List of List of POS tuples, 
#       "parse_trees": List of Parse Trees, 
#       "np_chunks": List of NP Chunks, 
#       "nes": List of Named Entities (3 types), 
#       "nes7": List of Named Entities (7 types)
#     },
#     "leadpara" : { same as before },
#     and so on
#   }

import nltk, glob, re, json, cPickle as pickle, os.path, subprocess
from lxml import etree
from lxml import html
import Chunker
import QuestionClassifier

# Parse nice files for each document
# docs = list of {docno, title, leadpara, text}

def clean_text(text):
  text = re.sub(r'<[a-zA-Z\/][^>]*>', '', text) # Remove XML/HTML Tags
  text = " ".join(text.split())
  text = re.sub(r'\[[^\[\]]*\]', '', text) # Remove square brackets
  text = re.sub(r"([\s]*\.[\s]*\.[\s]*|[\s]*\.[\s]*;[\s]*|[\s]*;[\s]*\.[\s]*|[\s]*\.[\s]*\.[\s]*\.[\s]*|[\s]*\.[\s]*\.[\s]*\.[\s]*\.[\s]*)", ". ", text)
  return text

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
        docno = tree.xpath("//DOCNO")[0].text.strip()
        if len(tree.xpath("//HEADLINE")) > 0:
          headline = clean_text(etree.tostring(tree.xpath("//HEADLINE")[0]))
        else:
          headline = None
        if len(tree.xpath("//LEADPARA")) > 0:
          leadpara = clean_text(etree.tostring(tree.xpath("//LEADPARA")[0]))
        elif len(tree.xpath("//LP")) > 0:
          leadpara = clean_text(etree.tostring(tree.xpath("//LP")[0]))
        else:
          leadpara = None
        if len(tree.xpath("//TEXT")) > 0:
          text = clean_text(etree.tostring(tree.xpath("//TEXT")[0]))
        else:
          text = None
        docs.append({ "docno": docno, "headline": headline, "leadpara": leadpara, "text": text})
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
  print "Parsing Questions..."
  parsed_questions = {}
  with open('data/train/questions.txt', 'r') as f:
    data = f.read()
    questions = re.split('[\s]*</top>[\s]*', data)
    assert len(questions) == 200
    questions.pop()
    qc = QuestionClassifier.QuestionClassifier()
    for question in questions:
      question_number = int(re.search(r"<num>[\s]*Number:[\s]*([0-9]+)", question).group(1))
      question = re.search(r"<desc>[\s]*Description:[\s]*([a-zA-Z0-9\-\?\'\. ]+)", question).group(1)
      question_words = nltk.word_tokenize(question)
      question_pos = nltk.pos_tag(question_words)
      question_nes = nltk.ne_chunk(question_pos)
      question_tree = Chunker.chunker.parse(question_pos)
      question_classification = qc.classify(question)
      qwords, nouns, nes = [], [], []
      for part in question_nes:
        try:
          nes.append((part.node, part.leaves()[0][0]))
        except:
          if part[1] == 'WP' or part[1] == 'WRB':
            qwords.append(part[0])
          elif part[1] == 'NN' or part[1] == 'NNP':
            nouns.append(part[0])
      # print qwords, nouns, nes
      # print question_pos
      parsed_questions[question_number] = { "question": question, "pos": question_pos, "ne": question_nes, "parse_tree": question_tree, "question_classification": question_classification, "question_words": qwords, "nouns": nouns, "ne_words": nes }
  with open('data/train/parsed_questions.txt', 'wb') as f:
    pickle.dump(parsed_questions, f)

# TODO Parse out more answers
def parse_answers():
  print "Parsing Answers..."
  parsed_answers = {}
  with open('data/train/answers.txt', 'r') as f:
    data = f.read()
    answers = data.split("Question")
    answers.pop(0)
    for answer in answers:
      parts = [a.strip() for a in answer.strip().split("\n")]
      qno = int(parts.pop(0))
      question = parts.pop(0)
      doc = [parts.pop(0)]
      theanswer = [parts.pop(0)]
      for part in parts:
        if re.match(r"^[A-Z0-9]+\-[A-Z0-9]+$", part):
          doc.append(doc)
        else:
          theanswer.append(part)
      parsed_answers[qno] = { "question": question, "docnos": doc, "answers": theanswer }
  with open('data/train/parsed_answers.txt', 'wb') as f:
    pickle.dump(parsed_answers, f)

def generate_parse_trees():
  print "Generating Parse Trees using the Stanford Parser..."
  try:
    os.makedirs('data/train/parsed_docs_trees')
  except:
    pass
  for filename in glob.glob("data/train/parsed_docs/top_docs.*"):
    if int(filename.rsplit(".", 1)[1]) <= 210:
      continue
    print filename
    pdocs = []
    with open(filename, 'rb') as f:
      docs = pickle.load(f)
      for docindex, doc in enumerate(docs):
        print doc['docno']
        pdoc = {}
        with open('temp.txt', 'w') as f:
          for k in ['headline', 'leadpara', 'text']:
            if doc[k] is not None:
              f.write(doc[k]+"\n")
        sentences, pos_tags, parse_trees, np_chunks, nes, nes7 = [], [], [], [], [], []
        # Get NEs (3 types)
        lines = subprocess.check_output(["../../../tools/stanford-ner-2012-04-07/ner.sh", "temp.txt"])
        lines = lines.strip()
        for i, l in enumerate(lines.split("\n")):
          ne = [tuple(w.split("/")) for w in l.split(" ") if len(w) > 0]
          if len(ne) > 100: # attempt to split
            temp_ne = []
            for n in ne:
              if (n[0] == ';' or n[0] == '.') and len(temp_ne) > 0:
                temp_ne.append(n)
                nes.append(temp_ne)
                temp_ne = []
              else:
                temp_ne.append(n)
            if len(temp_ne) > 0:
              nes.append(temp_ne)
          else:
            nes.append(ne)
        # Get NEs (7 types)
        lines = subprocess.check_output(["../../../tools/stanford-ner-2012-04-07/nes.sh", "temp.txt"])
        lines = lines.strip()
        for i, l in enumerate(lines.split("\n")):
          ne = [tuple(w.split("/")) for w in l.split(" ") if len(w) > 0]
          if len(ne) > 100: # attempt to split
            temp_ne = []
            for n in ne:
              if (n[0] == ';' or n[0] == '.') and len(temp_ne) > 0:
                temp_ne.append(n)
                nes7.append(temp_ne)
                temp_ne = []
              else:
                temp_ne.append(n)
            if len(temp_ne) > 0:
              nes7.append(temp_ne)
          else:
            nes7.append(ne)
        # Write tokenized strings to file again so that the NE and Parse Tree tokenization match up exactly
        with open('temp.txt', 'w') as f:
          for ne in nes:
            f.write(" ".join([w[0] for w in ne])+"\n")
        lines = subprocess.check_output(["../../../tools/stanford-parser-2012-03-09/parsetree.sh", "temp.txt"])
        lines = lines.strip()
        # Get POS Tags and Parse Trees
        for i, l in enumerate(lines.split("\n")):
          if i % 3 == 0: # POS Tags
            pos = [tuple(w.split("/")) for w in l.split(" ")]
            pos_tags.append(pos)
            sentences.append([w[0] for w in pos])
            chunks = Chunker.chunker.parse(pos)
            np_chunks.append(chunks)
          elif i % 3 == 2: # Parse Tree
            tree = nltk.tree.Tree.parse(l)
            parse_trees.append(tree)
        # Make sure everything matches up
        try:
          assert len(pos_tags) == len(parse_trees) == len(nes) == len(nes7)
        except:
          print doc['docno'], len(pos_tags), len(nes)
          for i, pos_tag in enumerate(pos_tags):
            print i, " ".join([w[0] for w in pos_tag])
          for i, ne in enumerate(nes):
            print i, " ".join([w[0] for w in ne])
          raise Exception
        # Okay, add to list of parsed documents
        pdocs.append({ "docno": doc['docno'], "sentences": sentences, "pos_tags": pos_tags, "parse_trees": parse_trees, "np_chunks": np_chunks, "nes": nes, "nes7": nes7 })
    with open('data/train/parsed_docs_trees/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(pdocs, f)

if __name__ == '__main__':
  # parse_docs()
  # parse_questions()
  parse_answers()
  # generate_parse_trees()
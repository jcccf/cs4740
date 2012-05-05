# -*- coding: utf-8 -*-
# Parser to parse documents, questions

# Useful formats you need to know
# /parsed_docs
#   List of { "docno": docno, "title": List of Strings, "leadpara": List of Strings, "text": List of Strings}
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
# /parsed_answers.txt
#   Dict of Question Number : { 
#       "question": question text, 
#       "docnos": list of document ids,
#       "answers": list of answer strings
#   }

import nltk, glob, re, json, cPickle as pickle, os.path, subprocess
from lxml import etree
from lxml import html
import Chunker
import QuestionClassifier
import argparse
import Loader

DIR = Loader.DIR

# Parse nice files for each document
# docs = list of {docno, title, leadpara, text}

def clean_text(text):
  text = re.sub(r'<[a-zA-Z\/][^>]*>', '', text) # Remove XML/HTML Tags
  text = " ".join(text.split())
  text = re.sub(r'\[[^\[\]]*\]', '', text) # Remove square brackets
  text = re.sub(r"([\-]+[ ]*){2,}", "-", text) # Turn multiple dashes into single dash
  text = re.sub(r"(\\|\|)", "", text)
  text = text.replace("`", "\'").replace("_", " ").replace("''", "\"")
  text = re.sub(r"([0-9]+)[\s]*\/[\s]*([0-9]+)[\s]*\/[\s]*([0-9]*)", r"\1\/\2\/\3", text)
  text = re.sub(r"([\'0-9]+)[\s]*\/[\s]*([\'0-9]+)", r"\1\/\2", text)
  text = re.sub(r"([\s]*\.[\s]*\.[\s]*|[\s]*\.[\s]*;[\s]*|[\s]*;[\s]*\.[\s]*|[\s]*\.[\s]*\.[\s]*\.[\s]*|[\s]*\.[\s]*\.[\s]*\.[\s]*\.[\s]*)", ". ", text)
  text = split_into_paras(text)
  return text

def clean_para(text):
  text = text.strip()
  text = re.sub(r'("(?=\S)[^"]*(?<=\S)")|"', lambda m: m.group(1) or '', text) # Clean unbalanced "
  text = re.sub(r'(\'(?=\S)[^\']*(?<=\S)\')|\'', lambda m: m.group(1) or '', text)
  text = re.sub(r'(\((?=\S)[^\(\)]*(?<=\S)\))|\(|\)', lambda m: m.group(1) or '', text) # Clean unbalanced ()
  text = re.sub(r'(\[(?=\S)[^\[\]]*(?<=\S)\])|\[|\]', lambda m: m.group(1) or '', text) # Clean unbalanced []
  text = re.sub(r'^[\s]*\;[\s]*(.+)', r'\1', text) # Remove ; from beginning of string
  return text

# Split into paragraphs containing no more than limit=100 words each.
# If a sentence contains more than 100 words, split it up.
def split_into_paras(text, limit=100):
  paragraphs = []
  sentences = nltk.sent_tokenize(text)
  paragraph, para_count = [], 0 
  actual_sentences = []
  # Generate sentence word counts, splitting long sentences if necessary
  for sentence in sentences:
    num_words = len(nltk.word_tokenize(sentence))
    if num_words > limit:
      subsentences = re.split(r"(;|--|\/|:)", sentence)
      for subsentence in subsentences:
        num_words = len(nltk.word_tokenize(subsentence))
        if num_words < limit:
          actual_sentences.append((subsentence, num_words))
        else:
          print "Skipped", subsentence
    else:
      if len(sentence.split(";")) > 4:
        sentence = sentence.replace(";", ".")
      actual_sentences.append((sentence, num_words))
  # Combine sentences into paragraphs
  for sentence, num_words in actual_sentences:
    if para_count + num_words < 100:
      paragraph.append(sentence)
      para_count += num_words
    else:
      if len(paragraph) > 0:
        paragraphs.append(" ".join(paragraph))
      paragraph, para_count = [sentence], num_words
  if len(paragraph) > 0:
    paragraphs.append(" ".join(paragraph))
  paragraphs = [clean_para(p) for p in paragraphs]
  return paragraphs
  
def parse_docs():
  print "Parsing Docs"
  try:
    os.makedirs(DIR+'/parsed_docs')
  except:
    pass

  parser = etree.XMLParser(recover=True)
  for filename in glob.glob(DIR+"/docs/top_docs.*"):
    print filename
    docs = []
    with open(filename, 'r') as f:
      data = f.read()
      xmldocs = re.split('[\s]*Qid:[\s]*[0-9]+[\s]*Rank:[\s]*[0-9]+[\s]*Score:[\s]*([0-9\.]+)[\s]*', data)
      assert len(xmldocs) == 101
      xmldocs = [ x.strip() for x in xmldocs if len(x.strip()) > 0 ]
      xmldocs = zip(xmldocs[0:len(xmldocs):2],xmldocs[1:len(xmldocs):2])
      for score,xmldoc in xmldocs:
        # print score,xmldoc
        # exit(0)
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
          text = []
          for elt in tree.xpath("//TEXT"):
            text.extend(clean_text(etree.tostring(elt)))
        else:
          text = None
        docs.append({ "docno": docno, "score": float(score), "headline": headline, "leadpara": leadpara, "text": text})
    # Write to a pickle
    with open(DIR+'/parsed_docs/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(docs, f)

# Identify the question type and parse out the NEs
def parse_questions():
  print "Parsing Questions..."
  parsed_questions = {}
  with open(DIR+'/questions.txt', 'r') as f:
    data = f.read()
    questions = re.split('[\s]*</top>[\s]*', data)
    if len(questions[-1].strip()) == 0: questions.pop()
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
  with open(DIR+'/parsed_questions.txt', 'wb') as f:
    pickle.dump(parsed_questions, f)

# Parse out answers
def parse_answers():
  print "Parsing Answers..."
  parsed_answers = {}
  with open(DIR+'/answers.txt', 'r') as f:
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
          doc.append(part)
        else:
          theanswer.append(part)
      parsed_answers[qno] = { "question": question, "docnos": doc, "answers": theanswer }
  with open(DIR+'/parsed_answers.txt', 'wb') as f:
    pickle.dump(parsed_answers, f)

def parse_real_answers_in_docs():
  answers = Loader.answers()
  found_answers = {}
  for i in range(201, 400):
    found_answers[i] = { 'question': answers[i]['question'], 'answers': [], 'docnos': []}
    docs = Loader.docs(i)
    for answer in answers[i]['answers']:
      found_docs = []
      for d_idx, d in enumerate(docs):
        if d['text'] is not None:
          for para in d['text']:
            if answer.lower() in para.lower():
              found_docs.append((d_idx, d['docno']))
      if len(found_docs) > 0:
        found_answers[i]['answers'].append(answer)
        found_answers[i]['docnos'].append(found_docs)
  with open(DIR+'/parsed_real_answers.txt', 'wb') as f:
    pickle.dump(found_answers, f)

if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-d', action='store_true', dest="docs", help="parse documents")
  argparser.add_argument('-q', action='store_true', dest="questions", help="parse questions")
  argparser.add_argument('-a', action='store_true', dest="answers", help="parse answers")
  argparser.add_argument('-r', action='store_true', dest="real_answers", help="parse real answers")
  args = argparser.parse_args()
  if args.docs:
    parse_docs()
  if args.questions:
    parse_questions()
  if args.answers:
    parse_answers()
  if args.real_answers:
    parse_real_answers_in_docs()

# -*- coding: utf-8 -*-
#
# A Possible Alternative Baseline
#

import nltk, re, cPickle as pickle

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
      parsed_questions[question_number] = { "number": question_number, "question": question, "raw_pos": question_pos, "raw_ne": question_nes, "qwords": qwords, "nouns": nouns, "nes": nes }
  with open('data/train/parsed_questions.txt', 'wb') as f:
    pickle.dump(parsed_questions, f)


# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def answer_all(answerer=naive_answer):
  with open('data/train/parsed_questions.txt', 'rb') as f:
    questions = pickle.load(f)
  for question in questions:
    with open('data/train/parsed_docs/top_docs.%d' % question['number'], 'rb') as f, open('data/train/parsed_docs_posne/top_docs.%d' % question['number'], 'rb') as f2:
      docs, docs_posne = f.read(), f2.read()
    answer = answerer(question, docs, docs_posne)


# Get POS tags/NEs for these sentences, and use the tags to identify the answer depending on the question type
#   For example, if the question is "How much...", look for the nearest number
def naive_answer(question, docs, docs_posne):
  # (Who/what) (is/was) (NE) ? => Find a description of NE
  if ("Who" in question['qwords'] or "What" in question['qwords']) and len(nes) == 1:
    # TODO Add code here!
    pass
  return None

if __name__ == '__main__':
  parse_questions()
  answer_all(answerer=naive_answer)
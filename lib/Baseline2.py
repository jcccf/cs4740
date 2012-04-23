# -*- coding: utf-8 -*-
#
# A Possible Alternative Baseline
#

import nltk, re, cPickle as pickle

# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def answer_all(answerer=naive_answer):
  with open('data/train/parsed_questions.txt', 'rb') as f:
    questions = pickle.load(f)
  for qno, question in questions.iteritems():
    with open('data/train/parsed_docs/top_docs.%d' % qno, 'rb') as f, open('data/train/parsed_docs_posne/top_docs.%d' % qno, 'rb') as f2:
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
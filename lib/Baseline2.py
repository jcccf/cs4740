# -*- coding: utf-8 -*-
#
# A Possible Alternative Baseline
#

import nltk, re, cPickle as pickle

# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def answer_all(answerer):
  with open('data/train/parsed_questions.txt', 'rb') as f:
    questions = pickle.load(f)
  for qno, question in questions.iteritems():
      # with open('data/train/parsed_docs/top_docs.%d' % qno, 'rb') as f, open('data/train/parsed_docs_posne/top_docs.%d' % qno, 'rb') as f2:
      #   docs, docs_posne = f.read(), f2.read()
      with open('data/train/parsed_docs/top_docs.%d' % qno, 'rb') as f:
        docs = f.read()
      answer = answerer(question, docs, None)

# Get POS tags/NEs for these sentences, and use the tags to identify the answer depending on the question type
#   For example, if the question is "How much...", look for the nearest number
def naive_answer(question, docs, docs_posne):
  # (Who/what) (is/was) (NE) ? => Find a description of NE
  if ("Who" in question['qwords'] or "What" in question['qwords']) and len(question['nes']) == 1:
    search_term = question['nes'][0][1]
    print question['question'], question['qwords'], question['nes']
  return None

if __name__ == '__main__':
  answer_all(answerer=naive_answer)
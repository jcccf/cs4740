# -*- coding: utf-8 -*-
#
# A Possible Alternative Baseline
#

import nltk, re, cPickle as pickle
import QuestionParser, DocFilterer, Loader

# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def answer_all(answerer):
  with open('data/naive_out.txt', 'w') as fout:
    questions = Loader.questions()
    for qno, question in questions.iteritems():
      print qno
      docs_posne = Loader.docs_posne(qno)
      docs = Loader.docs(qno)
      answer = answerer(question, docs, docs_posne)
      for ans in answer[:5]:
        fout.write("%d %s\n" % (qno, ans))

# Search for noun phrases in each sentence in each document, rank documents by # of matched keywords in noun phrases
# Chunk other noun phrases in each sentence and return as answer
def naive_answer(question, docs, docs_posne):
  # Get keywords from question
  keywords = QuestionParser.extract_keywords(question['parse_tree'])
  # print keywords

  # Find sentences that contain keywords, and sort by # of keyword matches
  all_matches = []
  for i, doc in enumerate(docs_posne):
    if doc['text'] is not None:
      matches = DocFilterer.naive_filter_sentences(keywords, doc['text']['sentences'])
      matches_with_index = [(i, s, c) for s, c in matches]
      all_matches += matches_with_index
  all_matches = sorted(all_matches, key=lambda x: -x[2])
  
  # Extract chunked NPs from sentences
  output_lines = []
  for i, s, c in all_matches:
    # print " ".join(docs_posne[i]['text']['sentences'][s]) # Print actual sentence found
    nps = DocFilterer.naive_extract_nps(keywords, docs_posne[i]['text']['parse_tree'][s])
    for chunk in DocFilterer.nps_to_chunks(nps):
      output_lines.append("%s %s" % (docs[i]['docno'], chunk))
  return output_lines

if __name__ == '__main__':
  answer_all(answerer=naive_answer)
  
  # # (Who/what) (is/was) (NE) ? => Find a description of NE
  # if (("Who is" in question['question'] or "Who was" in question['question']) and len(question['pos'])) == 1:
  #   print question['question'], question['question_words'], question['ne_words']
  # return None
# -*- coding: utf-8 -*-
#
# A Possible Alternative Baseline
#

import nltk, re, cPickle as pickle
import QuestionParser, DocFilterer, Loader
from CoreNLPLoader import *

# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def answer_all(answerer, use_chunk=False):
  with open('data/naive_out.txt', 'w') as fout:
    questions = Loader.questions()
    for qno, question in questions.iteritems():
      docs = CoreNLPLoader(qno)  
      print qno
      answer = answerer(question, docs, use_chunk)
      print answer
      if answer == None:
        fout.write("%d top_docs.%d nil\n" % (qno, qno))
      else:
        for ans in answer[:5]:
          fout.write("%d top_docs.%d %s\n" % (qno, qno, ans))

# Search for noun phrases in each sentence in each document, rank documents by # of matched keywords in noun phrases
# Chunk other noun phrases in each sentence and return as answer
def naive_answer(question, docs, use_chunk=False, doc_limit=50):
  # Get keywords from question
  keywords = QuestionParser.extract_keywords(question['parse_tree'])
  all_matches = []
  for doc_idx in range(0, min(doc_limit,len(docs.docs)) ):
    # Loop through each document
    paragraphs = docs.load_paras(doc_idx)
    # paragraphs = list of CoreNLPFeatures
    for paragraph_idx,paragraph in enumerate(paragraphs):
      tokenized_sentences = paragraph.tokenized()
      matches = DocFilterer.naive_filter_sentences(keywords, tokenized_sentences)
      matches_with_index = [(doc_idx, paragraph_idx, s, c) for s, c in matches]
      all_matches += matches_with_index
  all_matches = sorted(all_matches, key=lambda x: -x[3])
  # Extract chunked NPs from sentences
  output_lines = []
  for doc_idx, paragraph_idx, s, c in all_matches:
    paragraphs = docs.load_paras(doc_idx)
    paragraph = paragraphs[paragraph_idx]
    sentence_parse_tree = paragraph.parse_trees(flatten=True)[s]
    nps = DocFilterer.naive_extract_nps(keywords, sentence_parse_tree)                                      
    if (use_chunk):
      return DocFilterer.nps_to_chunks(nps)
    else:
      output = []
      for np in nps:
          answer = ""
          for w in np:
              answer += w[0]+" "
          output.append(answer.strip())
      return output

if __name__ == '__main__':
  answer_all(answerer=naive_answer, use_chunk=False)
  
  # # (Who/what) (is/was) (NE) ? => Find a description of NE
  # if (("Who is" in question['question'] or "Who was" in question['question']) and len(question['pos'])) == 1:
  #   print question['question'], question['question_words'], question['ne_words']
  # return None
import nltk, re, cPickle as pickle
import QuestionParser, DocFilterer, Loader

# Helper method that calls the answerer method on all questions
# answerer should have the signature answerer(question_dict, doclist, doc_posne_list)
def question_loader():
  questions = Loader.questions()
  answers = Loader.answers()
  for qno, question in questions.iteritems():
    print qno
    docs_posne = Loader.docs_posne(qno)
    docs = Loader.docs(qno)
    lines, docnos = search(question, docs, docs_posne)
    evaluate([w.lower() for w in answers[int(qno)]['answers'][0].split()], lines)
    count = 0
    for docno in answers[int(qno)]['docnos']:
      if docno in docnos:
        count += 1
    print count

# Search for noun phrases in each sentence in each document, rank documents by # of matched keywords in noun phrases
# Chunk other noun phrases in each sentence and return as answer
def search(question, docs, docs_posne):
  # Get keywords from question
  keywords = QuestionParser.extract_keywords(question['parse_tree'])

  # Find sentences that contain keywords, and sort by # of keyword matches
  all_matches = []
  for i, doc in enumerate(docs_posne):
    if doc['text'] is not None:
      matches = DocFilterer.naive_filter_sentences(keywords, doc['text']['sentences'])
      matches_with_index = [(i, s, c) for s, c in matches]
      all_matches += matches_with_index
  all_matches = sorted(all_matches, key=lambda x: -x[2])
  
  from collections import defaultdict
  output_lines, docnos = [], defaultdict(int)
  for i, s, c in all_matches:
    output_lines.append(docs_posne[i]['text']['sentences'][s])
    docnos[docs[i]['docno']] += c
  # print sorted(docnos.iteritems(), key=lambda x:-x[1])
  return (output_lines, docnos.keys())

def evaluate(answer, lines):
  matches = DocFilterer.naive_filter_sentences_unweighted(answer, lines)
  print matches

if __name__ == '__main__':
  question_loader()
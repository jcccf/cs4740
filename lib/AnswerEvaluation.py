import Loader, os, nltk
from CoreNLPLoader import *
from PipelineHelpers import *

def load_fake_answers(filename):  
  answers = {}
  with open(filename,"r") as f:
    for qno in range(201,400):
      anss = []
      for idx in range(5):
        line = f.readline()
        anss.append( line.split()[2:] )
      answers[qno] = anss
  return answers

def evaluate_answer(real_answers, answers):
  m_all_counts = []
  for real_answer in real_answers:
    keywords = nltk.word_tokenize(real_answer)
    m_counts = []
    for answer in answers:
      m_count = naive_filter_sentences_unweighted(keywords, [answer], False)[0][1]
      m_counts.append(m_count)
    m_all_counts.append((m_counts, len(keywords)))
  return m_all_counts

def evaluate_answers(filename):
  fake_answers = load_fake_answers(filename)
  real_answers = Loader.answers()
  for i in range(201, 400):
    real_answer = real_answers[i]
    m_all_counts = evaluate_answer(real_answer['answers'], fake_answers[i])
    
    correct = False
    for m_counts, keyword_len in m_all_counts:
      for m_count in m_counts:
        if float(m_count) >= 0.5 * keyword_len:
          correct = True
    if not correct:
      print i, m_all_counts
      
def answer_in_docs():
  answers = Loader.answers()
  for i in range(201, 400):  
    docs = Loader.docs(i)
    for d in docs:
      for answer in answers[i]:
        if d['text'] is not None:
          for para in d['text']:
            if answer.lower() in para.lower():
              print "FOUND ANSWER!"
    
  # For each answer in answers, search in documents for exact matches
  
  
if __name__ == '__main__':
  # evaluate_answers("wn_def_keywords_desc_qns.txt
  answer_in_docs()
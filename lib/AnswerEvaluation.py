import Loader, os, nltk, csv
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

def evaluate_answer(real_answers, fake_answers):
  m_all_counts = []
  for real_answer in real_answers:
    keywords = nltk.word_tokenize(real_answer)
    m_counts = []
    for fake_answer in fake_answers:
      m_count = naive_filter_sentences_unweighted(keywords, [fake_answer], False)[0][1]
      m_counts.append(m_count)
    m_all_counts.append((m_counts, len(keywords)))
  return m_all_counts

def evaluate_answers(filename):
  fake_answers = load_fake_answers(filename)
  real_answers = Loader.real_answers()
  questions = Loader.questions()
  # print len([True for ra in real_answers.itervalues() if len(ra['answers']) == 0])
  with open('answer_eval_%s' % filename, 'w') as f:
    writer = csv.writer(f)
    for i in range(201, 400):
      real_answer = real_answers[i]
      if len(real_answer['answers']) > 0:
        m_all_counts = evaluate_answer(real_answer['answers'], fake_answers[i])
      
        correct = False
        for m_counts, keyword_len in m_all_counts:
          for m_count in m_counts:
            if float(m_count) >= 1.0 * keyword_len:
              correct = True
        writer.writerow([i, questions[i]['question_classification'], 1 if correct else 0, m_all_counts])
        if not correct:
          print i, questions[i]['question_classification'], real_answer['question'], real_answer['answers'], m_all_counts
    
  # For each answer in answers, search in documents for exact matches
  
  
if __name__ == '__main__':
  evaluate_answers("wn_def_keywords_desc_qns.txt")
  # print answer_in_docs()
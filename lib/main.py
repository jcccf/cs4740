import WordParser, NGramModel, RandomSentence, AuthorPrediction, itertools
from math import exp

print "==CS 4740 Project 1=="
ngram = int(input("How many grams do you want? (unigram=1, bigram=2, trigram=3, etc.) "))
dataset = int(input("Which dataset do you want? (fbis=1, wsj=2, DataSet3=3, DataSet4=4, enron=5) "))

# DATASETS 1-4 RANDOM SENTENCE AND DOCUMENT PERPLEXITY
if dataset <= 4:
  if dataset == 1:
    train_file, test_file = 'data/fbis/fbis.train', 'data/fbis/fbis.test'
  elif dataset == 2:
    train_file, test_file = 'data/wsj/wsj.train', 'data/wsj/wsj.test'
  elif dataset == 3:
    train_file, test_file = 'data/Dataset3/Train.txt', 'data/Dataset3/Test.txt'
  elif dataset == 4:
    train_file, test_file = 'data/Dataset4/Train.txt', 'data/Dataset4/Test.txt'
  
  task = int(input("What task to perform? (random sentences=1, perplexity of test set=2) "))
  
  print "==Results=="
  
  cor = WordParser.WordParser(train_file)
  mod = NGramModel.NGramModel(ngram, smooth_type='lap') # TODO maybe not laplacian for all?
  mod.train([cor.words()])
  
  # RANDOM SENTENCE
  if task == 1:
    ran = RandomSentence.RandomSentence(mod)
    for i in range(10):
      print ' '.join(ran.gen_sentence(10))
  
  # PERPLEXITY
  elif task == 2:
    if dataset <= 2:
      print "Per-document Perplexity"
      test = WordParser.DocWordParser(test_file)
      for doc in test.docs():
        print mod.get_prob(list(itertools.chain.from_iterable(doc)))
    else:
      test = WordParser.WordParser(test_file)
      print mod.get_prob(test.words())

# ENRON
elif dataset == 5:
  print "==Results=="
  ap = AuthorPrediction.AuthorPrediction(ngram, smooth_type='lap')
  co = WordParser.EnronWordParser('data/EnronDataset/train.txt')
  co_test = WordParser.EnronWordParser('data/EnronDataset/test.txt')
  co_test_sentences = co_test.author_sentences()
  
  print "Loading authors...",
  for author, sentences in co.author_sentences().iteritems():
    ap.add_author(author, sentences)
  print "done"
  
  tp, fn, total = 0, 0, 0
  for author, sentences in co_test_sentences.iteritems():
    print "Predicting for %s..." % author
    for sentence in sentences:
      total += 1
      predicted = ap.predict_author(sentence)
      print predicted
      if predicted == author:
        tp += 1
  print "True Positives", tp
  print "Total", total
  print "Accuracy", (tp-0.0)/total

else:
  print "Invalid dataset chosen!"
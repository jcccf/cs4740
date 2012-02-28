import WordParser, NGramModel, RandomSentence, AuthorPrediction, itertools
from math import exp
import os

ngram_list = [1,2] # [1,2,3,4,5]
smoothing_list = ['lap', 'gte'] # ['none', 'lap']
unknown_list = ['first'] # ['none', 'first', 'once']
train_list = ['data/fbis/fbis.train', 'data/wsj/wsj.train', 'data/Dataset3/Train.txt', 'data/Dataset4/Train.txt']
test_list = ['data/fbis/fbis.test', 'data/wsj/wsj.test', 'data/Dataset3/Test.txt', 'data/Dataset4/Test.txt']
random_sentence_length = 100
stopping_punctuation = [".", "!"]

print "==CS 4740 Project 1=="
task = int(input("What task to perform? (random sentences=1, perplexity=2, author prediction=3) "))

# Random Sentences
if task == 1:
  try:
    os.makedirs('data/output/rand_sent/')
  except:
    pass

  for i, (train_file, test_file) in enumerate(zip(train_list, test_list)):
    if i <= 1:
      cor = WordParser.DocWordParser(train_file)
    else:
      cor = WordParser.WordParser(train_file)
    
    for smoothing_method in smoothing_list:
      for ngram_num in ngram_list:
        for unknown_method in unknown_list:
          mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method, unknown_type=unknown_method)
          mod.train([cor.words()])
          ran = RandomSentence.RandomSentence(mod)
          with open('data/output/rand_sent/%d_%d_%s_%s.txt' % (i+1, ngram_num, smoothing_method,unknown_method), 'w') as f:
            for j in range(10):
              sentence = ran.gen_sentence(random_sentence_length)
              first = len(sentence)
              for stop_word in stopping_punctuation:
                try:
                  first = min(first, sentence.index(stop_word)+1)
                  sentence = sentence[:first]
                except ValueError:
                  pass # no match
              f.write(' '.join(sentence) + '\n')

# Perplexity
elif task == 2:
  try:
    os.makedirs('data/output/perplexity/')
  except:
    pass

  for i, (train_file, test_file) in enumerate(zip(train_list, test_list)):
    if i <= 1:
      cor = WordParser.DocWordParser(train_file)
      test = WordParser.DocWordParser(test_file)
    else:
      cor = WordParser.WordParser(train_file)
      test = WordParser.WordParser(test_file)
    
    with open('data/output/perplexity/%d.txt' % (i+1), 'w') as f:
      for smoothing_method in smoothing_list:
        for ngram_num in ngram_list:
          for unknown_method in unknown_list:
            mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method, unknown_type=unknown_method)
            # mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method)
            mod.train([cor.words()])
            if i <= 1:
              # Per document perplexity
              with open('data/output/perplexity/%d_%d_%s_%s_doc.txt' % (i+1, ngram_num, smoothing_method,unknown_method), 'w') as f2:
              # with open('data/output/perplexity/%d_%d_%s_doc.txt' % (i+1, ngram_num, smoothing_method), 'w') as f2:
                for doc in test.docs():
                  f2.write('%f\n' % mod.get_perplexity(list(itertools.chain.from_iterable(doc))))
            f.write('%f %d %s\n' % (mod.get_perplexity(test.words()), ngram_num, smoothing_method))

# Enron Author Prediction
elif task == 3:
  try:
    os.makedirs('data/output/enron/')
  except:
    pass

  cor = WordParser.EnronWordParser('data/EnronDataset/train.txt')
  cor_val = WordParser.EnronWordParser('data/EnronDataset/validation.txt')
  cor_val_sentences = cor_val.author_sentence_tuples()
  cor_test = WordParser.EnronWordParser('data/EnronDataset/test.txt')
  cor_test_sentences = cor_test.author_sentence_tuples()
  
  for smoothing_method in smoothing_list:
    for ngram_num in ngram_list:
      for unknown_method in unknown_list:
        ap = AuthorPrediction.AuthorPrediction(ngram_num, smooth_type=smoothing_method, unknown_type=unknown_method)
          
        print "Loading authors...",
        for author, sentences in cor.author_sentences().iteritems():
          ap.add_author(author, sentences)
        print "done"
        
        total, tp = 0, 0

        with open('data/output/enron/%d_%s_%s_kaggle.txt' % (ngram_num, smoothing_method, unknown_method), 'w') as f2:
          with open('data/output/enron/%d_%s_%s.txt' % (ngram_num, smoothing_method, unknown_method), 'w') as f:
            for set_name, sentences in [('val',cor_val_sentences), ('test',cor_test_sentences)]:
              for author, sentence in sentences:
                values, predicted, rank = ap.predict_author(sentence, author)
                if set_name == 'val':
                  total += 1
                  if predicted == author:
                    tp += 1
                f.write('%s %s %f ' % (author, predicted, rank))
                for _,val in sorted(values.iteritems()):
                  f.write('%f ' % val)
                f.write('\n')
                f2.write('%s\n' % predicted)
        print 'Accuracy:', (tp-0.0)/total, tp, total
else:
  print "Invalid task chosen!"
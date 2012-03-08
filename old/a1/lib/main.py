import WordParser, NGramModel, RandomSentence, AuthorPrediction, itertools, os
from math import exp

#
# Parameters
#

# What type of n-grams to generate
ngram_list = [1] # Possible: [1,2,3,4,5, and so on]
# What kinds of smoothing to use
smoothing_list = ['none', 'lap', 'gte'] # Possible: ['none', 'lap', 'gte']
# What method of handling unknowns should be used?
unknown_list = ['none', 'first', 'once'] # Possible: ['none', 'first', 'once']

train_list = ['data/fbis/fbis.train', 'data/wsj/wsj.train', 'data/Dataset3/Train.txt', 'data/Dataset4/Train.txt']
test_list = ['data/fbis/fbis.test', 'data/wsj/wsj.test', 'data/Dataset3/Test.txt', 'data/Dataset4/Test.txt']
# train_list = ['data/wsj/wsj.train', 'data/Dataset4/Train.txt']
# test_list  = ['data/wsj/wsj.test', 'data/Dataset4/Test.txt']
random_sentence_length = 100
stopping_punctuation = [".", "!"]

#
# DO NOT MODIFY FROM HERE ON
#

print "==CS 4740 Project 1=="
task = int(input("What task to perform? (random sentences=1, perplexity=2, author prediction=3, kaggle=4) "))

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
          print "Training for %s, %d-gram, %s %s..." % (train_file, ngram_num, smoothing_method,unknown_method)
          mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method, unknown_type=unknown_method)
          mod.train(cor.docs_words())
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
    if i <= 0:
      cor = WordParser.DocWordParser(train_file)
      test = WordParser.DocWordParser(test_file)
    else:
      cor = WordParser.WordParser(train_file)
      test = WordParser.WordParser(test_file)
    
    with open('data/output/perplexity/%d.txt' % (i+1), 'w') as f:
      for smoothing_method in smoothing_list:
        for ngram_num in ngram_list:
          for unknown_method in unknown_list:
            print "Training for %s, %d-gram, %s %s..." % (train_file, ngram_num, smoothing_method,unknown_method)
            mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method, unknown_type=unknown_method)
            # mod = NGramModel.NGramModel(ngram_num, smooth_type=smoothing_method)
            mod.train([cor.words()])
            # mod.train(cor.sentences())
            if i <= 0:
              # Per document perplexity
              with open('data/output/perplexity/%d_%d_%s_%s_doc.txt' % (i+1, ngram_num, smoothing_method,unknown_method), 'w') as f2:
              # with open('data/output/perplexity/%d_%d_%s_doc.txt' % (i+1, ngram_num, smoothing_method), 'w') as f2:
                for doc in test.docs():
                  f2.write('%f\n' % mod.get_perplexity(list(itertools.chain.from_iterable(doc))))
            if unknown_method == "none":
              # n = cor.num_new_words(test.words())
              # mod.set_vocab_expansion(n)
              mod.expand_vocab(test.words())
              # print "%d vs %d"%(n,mod.vocab_expansion)
              # assert( n == mod.vocab_expansion )
            # mod.good_turing_discount_model()
            f.write('%f %d %s\n' % (mod.get_perplexity(test.words()), ngram_num, smoothing_method))
            mod.set_vocab_expansion(0)
            # mod.reset_dict()
            
            ####### cross perplexity computations ####
            for j, test_file in enumerate(test_list):
              if j <= 0:
                test2 = WordParser.DocWordParser(test_file)
              else:
                test2 = WordParser.WordParser(test_file)
              if unknown_method == "none":
                # n = cor.num_new_words(test2.words())
                # print "%s %s %d"%(train_file,test_file,n)
                # mod.set_vocab_expansion(n)
                mod.expand_vocab(test2.words())
                # assert( n == mod.vocab_expansion )
              # mod.good_turing_discount_model()
              f.write('cross%d %f %d %s %s\n' % (j+1, mod.get_perplexity(test2.words()), ngram_num, smoothing_method, unknown_method))
              mod.set_vocab_expansion(0)
              # mod.reset_dict()
            ####### cross perplexity computations ####  

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
        print "Training for %d-gram, %s %s..." % (ngram_num, smoothing_method,unknown_method)
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
        
# Kaggle Accuracy!
elif task == 4:
  cor_val = WordParser.EnronWordParser('data/EnronDataset/validation.txt')
  val_authors = [author for author, _ in cor_val.author_sentence_tuples()]
  test_authors = []
  with open('data/EnronDataset/test_solutions.txt', 'r') as f:
    for l in f:
      test_authors.append(l.strip())
  assert len(val_authors) == 2024
  assert len(test_authors) == 2024
  
  kag_files = []

  for dirname, dirnames, filenames in os.walk('data/Kaggle/'):
    for filename in filenames:
      fullpath = os.path.join(dirname, filename)
      if '_kaggle' in fullpath:
        kag_files.append((fullpath, filename))
  
  def num_correct(true, test):
    correct = 0
    for predicted, true in zip(true, test):
      if predicted == true:
        correct += 1
    return correct
    
  for full_path, filename in kag_files:
    with open(full_path, 'r') as f:
      lines = [l.strip() for l in f.readlines()]
      val_lines, test_lines = lines[:2024], lines[2024:]
      assert len(val_lines) == 2024 and len(test_lines) == 2024
      
      print "%s %d %d" % (filename, num_correct(val_authors, val_lines), num_correct(test_authors, test_lines))
    
else:
  print "Invalid task chosen!"
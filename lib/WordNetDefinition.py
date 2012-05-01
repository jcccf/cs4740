import nltk, re, cPickle as pickle, time
from nltk.corpus import wordnet as wn
from PipelineHelpers import remove_duplicates_list
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint

class Lemmatizer(object):
  def __init__(self):
    self.lemmatizer = WordNetLemmatizer()
    self.load()
    self.has_changed = False
    
  def lemmatize(self,word):
    word = word.lower()
    if word in self.cache:
      return self.cache[word]
    
    lemma = self.lemmatizer.lemmatize(word)
    self.cache[word] = lemma
    self.has_changed = True
    # print "Adding:",word,lemma
    return lemma
  
  def save(self,filename="data/train/cached_lemmas.txt"):
    if self.has_changed:
      with open(filename,"wb") as f:
        pickle.dump(self.cache, f)
        f.flush()
  
  def load(self,filename="data/train/cached_lemmas.txt"):
    try:
      with open(filename,"rb") as f:
        self.cache = pickle.load(f)
        # pprint(self.cache)
    except:
      self.cache = dict()

  def __del__(self):
    self.save()
    # time.sleep(1.0)
    # pass
    
lemmatizer = Lemmatizer()

def lemmatize(keywords,add_synsets=False):
  # lemmatizer = WordNetLemmatizer()
  new_keywords = []
  for word in keywords:
    new_keywords.append(lemmatizer.lemmatize(word))
    if add_synsets:
      synsets = wn.synsets(word)
      for syn in synsets:
        for lemma in syn.lemmas:
          new_keywords += lemma.name.lower().split("_")
  new_keywords = remove_duplicates_list(new_keywords)
  return new_keywords
  
def get_wordnet_def_entity(entity):
    synsets = wn.synsets(entity)
    if len(synsets) > 0:
        definition = synsets[0].definition
        return definition
    else:
        return None

def get_wordnet_def_entity_keywords(entity):
    synsets = wn.synsets(entity)
    if len(synsets) > 0:
      keywords = []
      for synset in synsets:
        for lemma in synset.lemmas:
          keywords.extend( lemma.name.lower().split("_") )
        definition = synset.definition.lower()
        tokenized_def = nltk.word_tokenize(definition)
        def_pos =  nltk.pos_tag(tokenized_def)
        for (word, pos) in def_pos:
            if (pos in ['NN','NNP','NNPS','NNS']) and word != '(': # for some reason nltk tags ( as NN
                keywords.append(word)
      return remove_duplicates_list(keywords)
    else:
      return None

def get_def_for_question_subject(question, output='keywords'):
    match = re.match('(What is a|What is) (\w+)?', question)
    entity = None
    if match != None:
        entity = match.groups()[1]
    match = re.match('(Who is|Who was) ([A-Z]\w+ [A-Z]\w+|[A-Z]\w+)?', question)
    if match != None:
        entity = match.groups()[1]
    match = re.match('(Where is|Where is the) ([A-Z]\w+ [A-Z]\w+|[A-Z]\w+).+', question)
    if match != None:
        entity = match.groups()[1]
    match = re.match('(What does|What does the abbreviation) (\w+) stand for?', question)
    if match != None:
        entity = match.groups()[1]
    if entity != None:
        if output == 'keywords':
            return get_wordnet_def_entity_keywords(entity)
        else:
            return get_wordnet_def_entity(entity)
    else:
        return None

if __name__ == '__main__':
    print get_def_for_question_subject('What is a caldera?', 'keywords')
    print get_def_for_question_subject('What is molybdenum?', 'keywords')
    print get_def_for_question_subject('Who was Picasso?', 'keywords')
    print get_def_for_question_subject('Where is Guam?', 'keywords')
    print get_def_for_question_subject('Where is the Kalahari desert?', 'keywords')
    print get_def_for_question_subject('Where is Romania located?', 'keywords')
    print get_def_for_question_subject('What does NAFTA stand for?', 'keywords')
    print get_def_for_question_subject('What does the abbreviation OAS stand for?', 'keywords')

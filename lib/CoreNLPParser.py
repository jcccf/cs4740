import jsonrpc, nltk
from nltk.tree import Tree
from simplejson import loads

# From NLTK Cookbook
def flatten_childtrees(trees):
  children = []
  for t in trees:
    if t.height() < 3:
      children.extend(t.pos())
    elif t.height() == 3:
      children.append(Tree(t.node, t.pos()))
    else:
      children.extend(flatten_childtrees([c for c in t]))
  return children
def flatten_deeptree(tree):
  return Tree(tree.node, flatten_childtrees([c for c in tree]))

# Class to contact CoreNLP server
class CoreNLPParser:
  def __init__(self,host="127.0.0.1",port=8080):
    self.server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=(host, port)))
    
  def parse(self, document_text):
    result = loads(self.server.parse(document_text))
    return result

# Class to parse JSON results from CoreNLP server
class CoreNLPFeatures:
  def __init__(self, result):
    self.result = result
  
  # Return a list of untokenized sentences
  def sentences(self):
    sentences = []
    for sentence in self.result['sentences']:
      sentences.append(sentence['text'])
    return sentences
  
  # Return a list of tokenized sentences
  def tokenized(self):
    sentences = []
    for sentence in self.result['sentences']:
      words = [w for w, a in sentence['words']]
      sentences.append(words)
    return sentences
    
  # Return a list of list of lemmas
  def lemmas(self):
    sentences = []
    for sentence in self.result['sentences']:
      lemmas = [a['Lemma'] for w, a in sentence['words']]
      sentences.append(lemmas)
    return sentences
  
  # Return a list of list of (word, pos) tuples
  def pos(self):
    pos_lines = []
    for sentence in self.result['sentences']:
      pos_line = [(w,a['PartOfSpeech']) for w, a in sentence['words']]
      pos_lines.append(pos_line)
    return pos_lines
    
  # Return a list of list of (List of words making up NE, NE type) tuples
  # Sentence -> Named Entity -> (Words in Named Entity, and type of the Named Entity)
  def named_entities(self):
    ne_list = []
    for sentence in self.result['sentences']:
      nes = []
      curr_ne, curr_ne_type = [], None
      for w, a in sentence['words']:
        if a['NamedEntityTag'] != "O":
          if curr_ne_type == a['NamedEntityTag']:
            curr_ne.append(w)
          else:
            if curr_ne_type is not None:
              nes.append((curr_ne, curr_ne_type))
            curr_ne, curr_ne_type = [w], a['NamedEntityTag']
        elif curr_ne_type is not None:
          nes.append((curr_ne, curr_ne_type))
          curr_ne, curr_ne_type = [], None
      if curr_ne_type is not None:
        nes.append((curr_ne, curr_ne_type))
      ne_list.append(nes)
    return ne_list
  
  # Return a list of NLTK parse trees
  # Reduce to only 1 sublevel if flatten is True
  def parse_trees(self, flatten=False):
    trees = []
    for sentence in self.result['sentences']:
      ptree = Tree.parse(sentence['parsetree'])
      if flatten:
        ptree = flatten_deeptree(ptree)
      trees.append(ptree)
    return trees
    
  # Return a list of list of tuples of (word(s), sentence_index, main_word_index?, start_entity_index, end_entity_index+1)
  # Each list is a coreference cluster
  # Else None if no coreferences
  # Coref Clusters -> Coref -> ( words:str, sent_idx in the same paragraph, main word idx?, start and end idx of entire coref words )
  def coreferences(self):
    if 'coref' in self.result:
      return self.result['coref']
    else:
      return None

if __name__ == '__main__':
  a = """
  The Golden Gate Bridge may soon undergo modern wind stress tests, partially to 
  determine how the 52-year-old span would withstand blustery weather if a second 
  deck is added for a rail transit system. A recommendation by chief bridge 
  engineer Dan Mohn to study how much wind it would take to damage the suspension 
  bridge, which in 1982 withstood 70-m.p.h. winds, was endorsed unanimously 
  Friday by a committee of bridge directors. 
  """
  # a = " ".join(a.split())
  # c = CoreNLPParser()
  # json = c.parse(a)
  # Sample result pasted below
  json = {'coref': [[[['it', 1, 3, 13, 14], ['The Golden Gate Bridge', 0, 3, 0, 4]]]], 'sentences': [{'parsetree': '(ROOT (S (NP (DT The) (NNP Golden) (NNP Gate) (NNP Bridge)) (VP (MD may) (ADVP (RB soon)) (VP (VB undergo) (NP (JJ modern) (NN wind) (NN stress) (NNS tests)) (, ,) (S (ADVP (RB partially)) (VP (TO to) (VP (VB determine) (SBAR (WHADVP (WRB how)) (S (NP (DT the) (JJ 52-year-old) (NN span)) (VP (MD would) (VP (VB withstand) (NP (JJ blustery) (NN weather)) (SBAR (IN if) (S (NP (DT a) (JJ second) (NN deck)) (VP (VBZ is) (VP (VBN added) (PP (IN for) (NP (DT a) (NN rail) (NN transit) (NN system)))))))))))))))) (. .)))', 'text': 'The Golden Gate Bridge may soon undergo modern wind stress tests, partially to determine how the 52-year-old span would withstand blustery weather if a second deck is added for a rail transit system.', 'tuples': [['det', 'Bridge', 'The'], ['nn', 'Bridge', 'Golden'], ['nn', 'Bridge', 'Gate'], ['nsubj', 'undergo', 'Bridge'], ['xsubj', 'determine', 'Bridge'], ['aux', 'undergo', 'may'], ['advmod', 'undergo', 'soon'], ['root', 'ROOT', 'undergo'], ['amod', 'tests', 'modern'], ['nn', 'tests', 'wind'], ['nn', 'tests', 'stress'], ['dobj', 'undergo', 'tests'], ['advmod', 'determine', 'partially'], ['aux', 'determine', 'to'], ['xcomp', 'undergo', 'determine'], ['advmod', 'withstand', 'how'], ['det', 'span', 'the'], ['amod', 'span', '52-year-old'], ['nsubj', 'withstand', 'span'], ['aux', 'withstand', 'would'], ['ccomp', 'determine', 'withstand'], ['amod', 'weather', 'blustery'], ['dobj', 'withstand', 'weather'], ['mark', 'added', 'if'], ['det', 'deck', 'a'], ['amod', 'deck', 'second'], ['nsubjpass', 'added', 'deck'], ['auxpass', 'added', 'is'], ['advcl', 'withstand', 'added'], ['det', 'system', 'a'], ['nn', 'system', 'rail'], ['nn', 'system', 'transit'], ['prep_for', 'added', 'system']], 'words': [['The', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '3', 'CharacterOffsetBegin': '0', 'PartOfSpeech': 'DT', 'Lemma': 'the'}], ['Golden', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '10', 'CharacterOffsetBegin': '4', 'PartOfSpeech': 'NNP', 'Lemma': 'Golden'}], ['Gate', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '15', 'CharacterOffsetBegin': '11', 'PartOfSpeech': 'NNP', 'Lemma': 'Gate'}], ['Bridge', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '22', 'CharacterOffsetBegin': '16', 'PartOfSpeech': 'NNP', 'Lemma': 'Bridge'}], ['may', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '26', 'CharacterOffsetBegin': '23', 'PartOfSpeech': 'MD', 'Lemma': 'may'}], ['soon', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '31', 'CharacterOffsetBegin': '27', 'PartOfSpeech': 'RB', 'Lemma': 'soon'}], ['undergo', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '39', 'CharacterOffsetBegin': '32', 'PartOfSpeech': 'VB', 'Lemma': 'undergo'}], ['modern', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '46', 'CharacterOffsetBegin': '40', 'PartOfSpeech': 'JJ', 'Lemma': 'modern'}], ['wind', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '51', 'CharacterOffsetBegin': '47', 'PartOfSpeech': 'NN', 'Lemma': 'wind'}], ['stress', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '58', 'CharacterOffsetBegin': '52', 'PartOfSpeech': 'NN', 'Lemma': 'stress'}], ['tests', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '64', 'CharacterOffsetBegin': '59', 'PartOfSpeech': 'NNS', 'Lemma': 'test'}], [',', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '65', 'CharacterOffsetBegin': '64', 'PartOfSpeech': ',', 'Lemma': ','}], ['partially', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '75', 'CharacterOffsetBegin': '66', 'PartOfSpeech': 'RB', 'Lemma': 'partially'}], ['to', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '78', 'CharacterOffsetBegin': '76', 'PartOfSpeech': 'TO', 'Lemma': 'to'}], ['determine', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '88', 'CharacterOffsetBegin': '79', 'PartOfSpeech': 'VB', 'Lemma': 'determine'}], ['how', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '92', 'CharacterOffsetBegin': '89', 'PartOfSpeech': 'WRB', 'Lemma': 'how'}], ['the', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '96', 'CharacterOffsetBegin': '93', 'PartOfSpeech': 'DT', 'Lemma': 'the'}], ['52-year-old', {'NormalizedNamedEntityTag': '52.0', 'Timex': '<TIMEX3 tid="t1" value="P52Y" type="DURATION">52-year-old</TIMEX3>', 'Lemma': '52-year-old', 'CharacterOffsetEnd': '108', 'PartOfSpeech': 'JJ', 'CharacterOffsetBegin': '97', 'NamedEntityTag': 'DURATION'}], ['span', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '113', 'CharacterOffsetBegin': '109', 'PartOfSpeech': 'NN', 'Lemma': 'span'}], ['would', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '119', 'CharacterOffsetBegin': '114', 'PartOfSpeech': 'MD', 'Lemma': 'would'}], ['withstand', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '129', 'CharacterOffsetBegin': '120', 'PartOfSpeech': 'VB', 'Lemma': 'withstand'}], ['blustery', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '138', 'CharacterOffsetBegin': '130', 'PartOfSpeech': 'JJ', 'Lemma': 'blustery'}], ['weather', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '146', 'CharacterOffsetBegin': '139', 'PartOfSpeech': 'NN', 'Lemma': 'weather'}], ['if', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '149', 'CharacterOffsetBegin': '147', 'PartOfSpeech': 'IN', 'Lemma': 'if'}], ['a', {'NormalizedNamedEntityTag': '2.0', 'Timex': '<TIMEX3 tid="t2" value="PT1S" type="DURATION">a second</TIMEX3>', 'Lemma': 'a', 'CharacterOffsetEnd': '151', 'PartOfSpeech': 'DT', 'CharacterOffsetBegin': '150', 'NamedEntityTag': 'DURATION'}], ['second', {'NormalizedNamedEntityTag': '2.0', 'Timex': '<TIMEX3 tid="t2" value="PT1S" type="DURATION">a second</TIMEX3>', 'Lemma': 'second', 'CharacterOffsetEnd': '158', 'PartOfSpeech': 'JJ', 'CharacterOffsetBegin': '152', 'NamedEntityTag': 'DURATION'}], ['deck', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '163', 'CharacterOffsetBegin': '159', 'PartOfSpeech': 'NN', 'Lemma': 'deck'}], ['is', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '166', 'CharacterOffsetBegin': '164', 'PartOfSpeech': 'VBZ', 'Lemma': 'be'}], ['added', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '172', 'CharacterOffsetBegin': '167', 'PartOfSpeech': 'VBN', 'Lemma': 'add'}], ['for', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '176', 'CharacterOffsetBegin': '173', 'PartOfSpeech': 'IN', 'Lemma': 'for'}], ['a', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '178', 'CharacterOffsetBegin': '177', 'PartOfSpeech': 'DT', 'Lemma': 'a'}], ['rail', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '183', 'CharacterOffsetBegin': '179', 'PartOfSpeech': 'NN', 'Lemma': 'rail'}], ['transit', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '191', 'CharacterOffsetBegin': '184', 'PartOfSpeech': 'NN', 'Lemma': 'transit'}], ['system', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '198', 'CharacterOffsetBegin': '192', 'PartOfSpeech': 'NN', 'Lemma': 'system'}], ['.', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '199', 'CharacterOffsetBegin': '198', 'PartOfSpeech': '.', 'Lemma': '.'}]]}, {'parsetree': '(ROOT (S (NP (NP (DT A) (NN recommendation)) (PP (IN by) (NP (NP (NN chief) (NN bridge) (NN engineer)) (SBAR (S (NP (NNP Dan) (NNP Mohn)) (VP (TO to) (VP (VB study) (SBAR (WHNP (WHADJP (WRB how) (JJ much)) (NN wind)) (S (NP (PRP it)) (VP (MD would) (VP (VB take) (S (VP (TO to) (VP (VB damage) (NP (DT the) (NN suspension) (NN bridge))))))))))))))) (, ,) (SBAR (WHNP (WDT which)) (S (PP (IN in) (NP (CD 1982))) (VP (VBD withstood) (NP (JJ 70-m.p.h.) (NNS winds))))) (, ,)) (VP (VBD was) (VP (VBN endorsed) (ADVP (RB unanimously)) (NP-TMP (NNP Friday)) (PP (IN by) (NP (NP (DT a) (NN committee)) (PP (IN of) (NP (NN bridge) (NNS directors))))))) (. .)))', 'text': 'A recommendation by chief bridge engineer Dan Mohn to study how much wind it would take to damage the suspension bridge, which in 1982 withstood 70-m.p.h. winds, was endorsed unanimously Friday by a committee of bridge directors.', 'tuples': [['det', 'recommendation', 'A'], ['nsubj', 'withstood', 'recommendation'], ['nsubjpass', 'endorsed', 'recommendation'], ['nn', 'engineer', 'chief'], ['nn', 'engineer', 'bridge'], ['prep_by', 'recommendation', 'engineer'], ['dobj', 'study', 'engineer'], ['nn', 'Mohn', 'Dan'], ['nsubj', 'study', 'Mohn'], ['aux', 'study', 'to'], ['infmod', 'engineer', 'study'], ['rcmod', 'engineer', 'study'], ['advmod', 'much', 'how'], ['amod', 'wind', 'much'], ['dobj', 'take', 'wind'], ['nsubj', 'take', 'it'], ['xsubj', 'damage', 'it'], ['aux', 'take', 'would'], ['ccomp', 'study', 'take'], ['aux', 'damage', 'to'], ['xcomp', 'take', 'damage'], ['det', 'bridge', 'the'], ['nn', 'bridge', 'suspension'], ['dobj', 'damage', 'bridge'], ['prep_in', 'withstood', '1982'], ['rcmod', 'recommendation', 'withstood'], ['amod', 'winds', '70-m.p.h.'], ['dobj', 'withstood', 'winds'], ['auxpass', 'endorsed', 'was'], ['root', 'ROOT', 'endorsed'], ['advmod', 'endorsed', 'unanimously'], ['tmod', 'endorsed', 'Friday'], ['det', 'committee', 'a'], ['agent', 'endorsed', 'committee'], ['nn', 'directors', 'bridge'], ['prep_of', 'committee', 'directors']], 'words': [['A', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '201', 'CharacterOffsetBegin': '200', 'PartOfSpeech': 'DT', 'Lemma': 'a'}], ['recommendation', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '216', 'CharacterOffsetBegin': '202', 'PartOfSpeech': 'NN', 'Lemma': 'recommendation'}], ['by', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '219', 'CharacterOffsetBegin': '217', 'PartOfSpeech': 'IN', 'Lemma': 'by'}], ['chief', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '225', 'CharacterOffsetBegin': '220', 'PartOfSpeech': 'NN', 'Lemma': 'chief'}], ['bridge', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '232', 'CharacterOffsetBegin': '226', 'PartOfSpeech': 'NN', 'Lemma': 'bridge'}], ['engineer', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '241', 'CharacterOffsetBegin': '233', 'PartOfSpeech': 'NN', 'Lemma': 'engineer'}], ['Dan', {'NamedEntityTag': 'PERSON', 'CharacterOffsetEnd': '245', 'CharacterOffsetBegin': '242', 'PartOfSpeech': 'NNP', 'Lemma': 'Dan'}], ['Mohn', {'NamedEntityTag': 'PERSON', 'CharacterOffsetEnd': '250', 'CharacterOffsetBegin': '246', 'PartOfSpeech': 'NNP', 'Lemma': 'Mohn'}], ['to', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '253', 'CharacterOffsetBegin': '251', 'PartOfSpeech': 'TO', 'Lemma': 'to'}], ['study', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '259', 'CharacterOffsetBegin': '254', 'PartOfSpeech': 'VB', 'Lemma': 'study'}], ['how', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '263', 'CharacterOffsetBegin': '260', 'PartOfSpeech': 'WRB', 'Lemma': 'how'}], ['much', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '268', 'CharacterOffsetBegin': '264', 'PartOfSpeech': 'JJ', 'Lemma': 'much'}], ['wind', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '273', 'CharacterOffsetBegin': '269', 'PartOfSpeech': 'NN', 'Lemma': 'wind'}], ['it', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '276', 'CharacterOffsetBegin': '274', 'PartOfSpeech': 'PRP', 'Lemma': 'it'}], ['would', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '282', 'CharacterOffsetBegin': '277', 'PartOfSpeech': 'MD', 'Lemma': 'would'}], ['take', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '287', 'CharacterOffsetBegin': '283', 'PartOfSpeech': 'VB', 'Lemma': 'take'}], ['to', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '290', 'CharacterOffsetBegin': '288', 'PartOfSpeech': 'TO', 'Lemma': 'to'}], ['damage', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '297', 'CharacterOffsetBegin': '291', 'PartOfSpeech': 'VB', 'Lemma': 'damage'}], ['the', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '301', 'CharacterOffsetBegin': '298', 'PartOfSpeech': 'DT', 'Lemma': 'the'}], ['suspension', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '312', 'CharacterOffsetBegin': '302', 'PartOfSpeech': 'NN', 'Lemma': 'suspension'}], ['bridge', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '319', 'CharacterOffsetBegin': '313', 'PartOfSpeech': 'NN', 'Lemma': 'bridge'}], [',', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '320', 'CharacterOffsetBegin': '319', 'PartOfSpeech': ',', 'Lemma': ','}], ['which', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '326', 'CharacterOffsetBegin': '321', 'PartOfSpeech': 'WDT', 'Lemma': 'which'}], ['in', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '329', 'CharacterOffsetBegin': '327', 'PartOfSpeech': 'IN', 'Lemma': 'in'}], ['1982', {'NormalizedNamedEntityTag': '1982', 'Timex': '<TIMEX3 tid="t1" value="1982" type="DATE">1982</TIMEX3>', 'Lemma': '1982', 'CharacterOffsetEnd': '334', 'PartOfSpeech': 'CD', 'CharacterOffsetBegin': '330', 'NamedEntityTag': 'DATE'}], ['withstood', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '344', 'CharacterOffsetBegin': '335', 'PartOfSpeech': 'VBD', 'Lemma': 'withstand'}], ['70-m.p.h.', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '354', 'CharacterOffsetBegin': '345', 'PartOfSpeech': 'JJ', 'Lemma': '70-m.p.h.'}], ['winds', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '360', 'CharacterOffsetBegin': '355', 'PartOfSpeech': 'NNS', 'Lemma': 'wind'}], [',', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '361', 'CharacterOffsetBegin': '360', 'PartOfSpeech': ',', 'Lemma': ','}], ['was', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '365', 'CharacterOffsetBegin': '362', 'PartOfSpeech': 'VBD', 'Lemma': 'be'}], ['endorsed', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '374', 'CharacterOffsetBegin': '366', 'PartOfSpeech': 'VBN', 'Lemma': 'endorse'}], ['unanimously', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '386', 'CharacterOffsetBegin': '375', 'PartOfSpeech': 'RB', 'Lemma': 'unanimously'}], ['Friday', {'NormalizedNamedEntityTag': 'XXXX-WXX-5', 'Timex': '<TIMEX3 tid="t2" value="XXXX-WXX-5" type="DATE">Friday</TIMEX3>', 'Lemma': 'Friday', 'CharacterOffsetEnd': '393', 'PartOfSpeech': 'NNP', 'CharacterOffsetBegin': '387', 'NamedEntityTag': 'DATE'}], ['by', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '396', 'CharacterOffsetBegin': '394', 'PartOfSpeech': 'IN', 'Lemma': 'by'}], ['a', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '398', 'CharacterOffsetBegin': '397', 'PartOfSpeech': 'DT', 'Lemma': 'a'}], ['committee', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '408', 'CharacterOffsetBegin': '399', 'PartOfSpeech': 'NN', 'Lemma': 'committee'}], ['of', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '411', 'CharacterOffsetBegin': '409', 'PartOfSpeech': 'IN', 'Lemma': 'of'}], ['bridge', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '418', 'CharacterOffsetBegin': '412', 'PartOfSpeech': 'NN', 'Lemma': 'bridge'}], ['directors', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '428', 'CharacterOffsetBegin': '419', 'PartOfSpeech': 'NNS', 'Lemma': 'director'}], ['.', {'NamedEntityTag': 'O', 'CharacterOffsetEnd': '429', 'CharacterOffsetBegin': '428', 'PartOfSpeech': '.', 'Lemma': '.'}]]}]}
  f = CoreNLPFeatures(json)
  print f.sentences()
  print f.tokenized()
  print f.pos()
  print f.lemmas()
  print f.named_entities()
  print f.parse_trees(flatten=True)
  print f.coreferences()
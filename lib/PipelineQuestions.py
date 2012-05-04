import nltk, Loader
from PipelineHelpers import *
from CoreNLPLoader import *

# Load all questions at once
class QuestionFeatures:
  def __init__(self):
    self.qs = Loader.questions()
    self.qs_core = CoreNLPQuestionLoader()
    
  # Return a dictionary of
  # question classification, named entities, nouns, keywords
  def features(self, qno):
    q_core = self.qs_core.load_question(qno)
    words = q_core.tokenized()[0]
    parse_tree = q_core.parse_trees(flatten=True)[0]
    keywords = extract_keywords(parse_tree)
    pos = q_core.pos()[0]
    nes = q_core.named_entities()[0]
    classification = self.qs[qno]['question_classification']
    return { "words": words, "keywords": keywords, "pos": pos, "nes": nes, "parse_tree": parse_tree, "classification": classification }

  def __str__(self):
    return str(self.features())
    
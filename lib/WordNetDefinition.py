import nltk, re
from nltk.corpus import wordnet as wn

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
        definition = synsets[0].definition
        tokenized_def = nltk.word_tokenize(definition)
        def_pos =  nltk.pos_tag(tokenized_def)
        keywords = []
        for (word, pos) in def_pos:
            if (pos in ['NN','NNP','NNPS','NNS']) and word != '(': # for some reason nltk tags ( as NN
                keywords.append(word)
        return keywords
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
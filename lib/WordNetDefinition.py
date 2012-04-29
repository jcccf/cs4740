import nltk, re
from nltk.corpus import wordnet as wn

def get_wordnet_def_entity(entity):
    synsets = wn.synsets(entity)
    if len(synsets) > 0:
        definition = synsets[0].definition
        return definition
    else:
        return None

def get_def_for_quesiton_subject(question):
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
        return get_wordnet_def_entity(entity)
    else:
        return None

if __name__ == '__main__':
    print get_def_for_quesiton_subject('What is a caldera?')
    print get_def_for_quesiton_subject('What is molybdenum?')
    print get_def_for_quesiton_subject('Who was Picasso?')
    print get_def_for_quesiton_subject('Where is Guam?')
    print get_def_for_quesiton_subject('Where is the Kalahari desert?')
    print get_def_for_quesiton_subject('Where is Romania located?')
    print get_def_for_quesiton_subject('What does NAFTA stand for?')
    print get_def_for_quesiton_subject('What does the abbreviation OAS stand for?')
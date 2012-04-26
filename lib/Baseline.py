import re, os
import xml.etree.ElementTree
import nltk

try:
    os.makedirs('data/output/')
except:
    pass

def parse_docs(question_number, NE_types, keywords):
    NE_dict = dict()
    text = ""
    keyword_match_positions = []
    answer_candidates = []
    at_text = False
    with open('data/train/docs/top_docs.%d' % question_number,'r') as f:
        #unfortunately, some of the articles have malformed XML, so xml parsing fails
        for line in f:
            if re.match('Qid: \d+', line) or (not line.strip()):
                continue 
            elif re.match('<TEXT>(.+)', line):
                text = re.match('<TEXT>(.+)', line).groups()[0]
                at_text = True
            elif re.match('(.+)</TEXT>', line):
                text += re.match('(.+)</TEXT>', line).groups()[0]
                at_text = False
                text_pos =  nltk.pos_tag(nltk.word_tokenize(text))
                chunk_tree = nltk.ne_chunk(text_pos, binary=False)
                word_position = 0
                for child in chunk_tree:
                    if (type(child) is nltk.Tree):
                        NE_string = child.pprint()
                        NE_string = NE_string.replace('\n', '').replace('  ', ' ')
                        entity_type, entity = re.match('\((\w+) ([^)]+)', NE_string).groups()
                        entity = entity.replace('/NNP', '').replace('/JJ', '')
                        if entity in keywords:
                            keyword_match_positions.append(word_position)
                        elif entity_type in NE_types:
                            answer_candidates.append((entity, word_position)) 
                    elif child[0] in keywords: #keyword match
                        keyword_match_positions.append(word_position)
                    word_position += 1
                for (candidate,candidate_pos) in answer_candidates:
                    if candidate in NE_dict:
                        NE_dict[candidate] += len(keyword_match_positions)
                    else:
                        NE_dict[candidate] = len(keyword_match_positions)
                                                                                
            elif at_text == True:
                text += line
    return NE_dict

# could probably perform better if a learning question classifier was used
def answer(question_num, question):
    question_pos =  nltk.pos_tag(nltk.word_tokenize(question))
    keywords = [];
    # get keywords from question
    for (word, pos) in question_pos:
        if pos == 'NN':
            keywords.append(word)
    if re.search(r'[wW]ho', question) != None:
        NE_dict = parse_docs(question_num, ["PERSON"], keywords)
    elif re.search('[wW]here', question) != None:
        NE_dict = parse_docs(question_num, ["GPE","LOCATION"], keywords)
    elif re.search('[wW]hat', question) != None:
        NE_dict = parse_docs(question_num, ["ORGANIZATION","GPE","FACILITY","GSP","LOCATION"], keywords)
    elif re.search('[wW]when', question) != None:
        NE_dict = parse_docs(question_num, ["ORGANIZATION","GPE","FACILITY","GSP","LOCATION"], keywords)
    elif re.search('[hH]ow', question) != None:
        NE_dict = parse_docs(question_num, ["ORGANIZATION","GPE","FACILITY","GSP","LOCATION"], keywords)
    else: # couldn't classify the question
        print question
        return None
    NE_list = sorted(NE_dict, key=lambda key: NE_dict[key])
    NE_list.reverse()
    return NE_list # returns list of NE in descending order by frequency

def answer_questions(filename='data/train/questions.txt'):
    question_num = 0
    with open(filename,'r') as f:
        with open('data/output/baseline_answers.txt','w') as fout:
            for line in f:
                if (not line.strip()) or re.match('<top>', line) or re.match('</top>', line) or re.match('<desc> Description:', line):
                    continue
                question_num_match = re.match('<num> Number: (\d+)', line)
                if question_num_match != None:
                    question_num = int(question_num_match.groups()[0])
                else:
                    NE_list = answer(question_num, line)                        
                    for i in range(5):
                        if (NE_list == None or i >= len(NE_list)):
                            fout.write("%d nil nil\n" % question_num)
                        else:
                            fout.write("%d top_docs.%d %s\n" % (question_num, question_num, NE_list[i]))
               
if __name__ == '__main__':
    answer_questions()
import re, os
import xml.etree.ElementTree
import nltk

try:
    os.makedirs('data/output/')
except:
    pass

def parse_docs(question_number, NE_types):
    NE_dict = dict()
    doc_found_at = dict()
    docno = ""
    text = ""
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
                for child in chunk_tree:
                    if (type(child) is nltk.Tree):
                        NE_string = child.pprint()
                        NE_string = NE_string.replace('\n', '').replace('  ', ' ')
                        entity_type, entity = re.match('\((\w+) ([^)]+)', NE_string).groups()
                        entity = entity.replace('/NNP', '').replace('/JJ', '')  
                        if entity_type in NE_types:
                            if entity in NE_dict:
                                NE_dict[entity] += 1
                            else:
                                NE_dict[entity] = 1
                            if entity not in doc_found_at:
                                doc_found_at[entity] = docno     
            elif at_text == True:
                text += line
    return (NE_dict,doc_found_at)

# could probably perform better if a learning question classifier was used
def answer(question_num, question):
    if re.search(r'[wW]ho', question) != None:
        (NE_dict,doc_found_at) = parse_docs(question_num, ["PERSON"])
    elif re.search('[wW]here', question) != None:
        (NE_dict,doc_found_at) = parse_docs(question_num, ["GPE"])
    elif re.search('[wW]hat', question) != None:
        (NE_dict,doc_found_at) = parse_docs(question_num, ["ORGANIZATION","GPE"])
    elif re.search('[wW]when', question) != None:
        # can't do dates with NE, need to find some other mechanism
        return (None, None)
    else: # couldn't classify the question
        return (None, None)
    NE_list = sorted(NE_dict, key=lambda key: NE_dict[key])
    NE_list.reverse()
    return (NE_list,doc_found_at) # returns list of NE in descending order by frequency

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
                    (NE_list,doc_found_at) = answer(question_num, line)                        
                    for i in range(5):
                        if (NE_list == None or i >= len(NE_list)):
                            fout.write("%d nil nil\n" % question_num)
                        else:
                            fout.write("%d %s %s\n" % (question_num, doc_found_at[NE_list[i]], NE_list[i]))
               
if __name__ == '__main__':
    #parse_docs(201, ["PERSON"])
    answer_questions()
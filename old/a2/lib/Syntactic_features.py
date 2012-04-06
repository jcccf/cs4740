import re

#prepare file for input to the Stanford parser
#returns a list of target words in the order that they are in inside the file
def prepare_file(filename):
    word_list = []
    line_matcher = re.compile(r'([\w]+)\.([\w]+) ([0-9 ]+)@[ ]*(.*)@([\w]+)@(.*)')
    with open(filename+'sout', 'w') as fout:
        with open(filename, 'r') as f:
            for l in f:
                match_obj = line_matcher.match(l)
                if match_obj:
                    word, pos, senses, context_before, target, context_after = match_obj.groups()
                    sentence_end_matcher = re.compile('[.?!]')
                    split_context_before = re.split('[.?!]', context_before)
                    split_context_after = re.split('[.?!]', context_after)
                    fout.write(split_context_before[-1].replace('@', '') + target + split_context_after[0].replace('@', '') +'.\n')
                    word_list.append(target)
                else:
                    raise Exception("Example Regex Failed to Match on\n%s" % l)
    return word_list

def parse_stanford_output(filename, word_list):
    line_matcher = re.compile(r'([\w]+)\(([^-]+)[-\d, ]+([^-]+).*')
    tree_matcher = re.compile('\(ROOT \(.*')
    word_list_index = -1
    synfeatures = []
    with open(filename+'sout'+'.stp', 'r') as f:
        for l in f:
            if (l == '\n' or l == ''):
                pass
            elif tree_matcher.match(l):
                word_list_index += 1 #next word
                synfeatures.append([])
                pass
            else:
                match_obj = line_matcher.match(l)
                if match_obj:
                    relation, word1, word2 = match_obj.groups()
                    if relation in ['prep_with','amod','prep_at','prep_of','det']:
                        #possibly use dobj
                        if word_list[word_list_index] == word1:
                            #synfeatures[word_list_index].append(relation)
                            synfeatures[word_list_index].append(word2) 
                        elif word_list[word_list_index] == word2:
                            #synfeatures[word_list_index].append(relation)
                            synfeatures[word_list_index].append(word1) 
                else:
                    pass
    return synfeatures

if __name__ == '__main__':
    filename = 'data/wsd-data/train_split.data';
    #filename = 'data/wsd-data/valiation_split.data';
    #filename = 'data/wsd-data/train.data';
    #filename = 'data/wsd-data/test.data';
    word_list = prepare_file(filename);
    #use the stanford parser to parse the output file
    raw_input("Parse the file with Stanford parser and press any key to continue")
    synfeatures = parse_stanford_output(filename, word_list);
    print synfeatures
               
      
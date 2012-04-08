        
def parse_opened_training_file(f):
    # Parses _opened_ file <f> and
    # returns a list of [ list of (POS, word) tuples ]
    data = []
    sentence = []
    for line in f:
        s = line.split()
        if len(s) != 2:
            print "ERROR:", line
            exit(-1)
        if s[0] == "<s>" and len(sentence) > 0:
            data.append(sentence)
            sentence = []
        sentence.append( tuple(s) )
    if len(sentence) > 0:
        data.append(sentence)
    return data
    
def parse_training_file(filename='data/pos_files/train.pos'):
    # Opens the file and passes it along
    with open(filename,'r') as f:
        return parse_opened_training_file(f)

def parse_opened_test_file(f):
    # Parses _opened_ file <f> and
    # returns a list of [ list of word ]
    data = []
    sentence = []
    for s in f:
        s = s.strip()
        if s == "<s>" and len(sentence) > 0:
            data.append(sentence)
            sentence = []
        sentence.append( s )
    if len(sentence) > 0:
        data.append(sentence)
    return data

def parse_test_file(filename='data/pos_files/test-obs.pos'):
    # Opens the file and passes it along
    with open(filename,'r') as f:
        return parse_opened_test_file(f)
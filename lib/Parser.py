        
def parse_training_file(filename='data/pos_files/train.pos'):
    data = []
    sentence = []
    with open(filename,'r') as f:
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

def parse_test_file(filename='data/pos_files/test-obs.pos'):
    data = []
    sentence = []
    with open(filename,'r') as f:
        for s in f:
            if s == "<s>" and len(sentence) > 0:
                data.append(sentence)
                sentence = []
            sentence.append( s )
    if len(sentence) > 0:
        data.append(sentence)
    return data

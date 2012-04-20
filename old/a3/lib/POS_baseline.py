
def train(filename='data/pos_files/train.pos'):
    tokens = dict()
    # count the occurrences of tags for each token in the training corpus
    with open(filename,'r') as f:
        for line in f:
            s = line.split()
            if len(s) != 2:
                print "ERROR:", line
                exit(-1)
            if s[1] not in tokens:
                tokens[s[1]] = dict() # dict within a dict
            tag_counts = tokens[s[1]]
            if s[0] in tag_counts:
                tag_counts[s[0]] += 1
            else:
                tag_counts[s[0]] = 1
            tokens[s[1]] = tag_counts
    # reduce to a dict with key=token and value=most frequent POS for that token
    model = dict()
    for (token, tag_counts) in tokens.iteritems():
        model[token] = (max(tag_counts.iteritems(), key=lambda tup : tup[1]))[0]
    return model

def tag_testdata(model, filename='data/pos_files/test-obs.pos'):
    with open(filename,'r') as f:
        with open('data/baselinepredictions.pos','w') as fout:
            for line in f:
                s = line.split()
                if len(s) != 1:
                    print "ERROR:", line
                    exit(-1)
                if s[0] == "<s>":
                    fout.write("<s> <s>\n")
                elif s[0] in model:
                    # tag with most frequent tag for that word
                    fout.write(model[s[0]]+" "+s[0]+"\n")
                else:
                    # word did not occur in training corpus
                    # tag it as NNP since it is likely to be a name
                    fout.write("NNP"+" "+s[0]+"\n")
    #print("done")
    
def validate(model, filename='data/pos_files/validation.pos'):
    wrong_pred = 0.0
    correct_pred = 0.0;
    with open(filename,'r') as f:
        for line in f:
            s = line.split()
            if len(s) != 2:
                print "ERROR:", line
                exit(-1)
            if s[0] == "<s>":
                pass
            else:
                if s[1] in model:
                    # tag with most frequent tag for that word
                    predict = model[s[1]]
                else:
                    # word did not occur in training corpus
                    # tag it as NNP since it is likely to be a name
                    predict = "NNP"
                if (predict == s[0]):
                    correct_pred += 1
                else:
                    wrong_pred += 1
    return (correct_pred/(correct_pred+wrong_pred))
            
if __name__ == '__main__':
    # tag the test data
    model = train('data/pos_files/train.pos')
    tag_testdata(model, 'data/pos_files/test-obs.pos')
    # validation
    #model = train('data/pos_files/train_split.pos')
    #accuracy = validate(model, 'data/pos_files/validation_split.pos')
    #print("Accuracy: %f" % accuracy)
    #accuracy = 0.0
    #for i in range(10):
    #    model = train('data/pos_files/train_split%d.pos' % i)
    #    accuracy += validate(model, 'data/pos_files/validation_split%d.pos' % i)
    #accuracy = accuracy / 10.0
    #print("Accuracy: %f" % accuracy)
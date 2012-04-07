def train(filename='data/ne_files/Reuters_NER_train.txt'):
    label_counts = dict()
    # count the occurrences of labels in the training corpus
    with open(filename,'r') as f:
        line_number = 0
        for line in f:
            line_number += 1
            if line_number % 3 == 0:             
                s = line.split()
                for label in s:
                    if label != "O":
                        if label in label_counts:
                            label_counts[label] += 1
                        else:
                            label_counts[label] = 1
    most_likely_tag = (max(label_counts.iteritems(), key=lambda tup : tup[1]))[0]
    return most_likely_tag

def tag_testdata(most_likely_tag, filename='data/ne_files/Reuters_NER_test.txt'):
    with open(filename,'r') as f:
        line_number = 0
        with open('data/ne_files/Reuters_NER_predict.txt','w') as fout:
            for line in f:
                line_number += 1
                if line_number % 2 == 1:
                    s = line.split()
                    predict = ""
                    for i in range(len(s)):
                        word = s[i]
                        if word[0].isupper():
                            # word begins with capital letter
                            predict += most_likely_tag
                        else:
                            predict += "O"
                        if i < len(s) - 1:
                            predict += " "
                    fout.write(line)
                if line_number % 2 == 0:
                    fout.write(line)
                    fout.write(predict+"\n")
    print("done")
                    
def validate(most_likely_tag, filename='data/ne_files/validation_split.txt'):
    tp = 0.0 #variables for computing F1 score
    fp = 0.0
    tn = 0.0
    fn = 0.0
    with open(filename,'r') as f:
        line_number = 0
        for line in f:
            line_number += 1
            if line_number % 3 == 1:
                s = line.split()
                predict = []
                for word in s:
                    if word[0].isupper():
                        # word begins with capital letter
                        predict.append(most_likely_tag)
                    else:
                        predict.append("O")
            if line_number % 3 == 0:
                s = line.split()
                for i in range(len(s)):
                    if predict[i] == "O":
                        if s[i] == "O":
                            tn += 1.0
                        else:
                            fn += 1.0
                    else: # system tagged word as a NE
                        if s[i] == predict[i]:
                            tp += 1.0
                        else:
                            fp += 1.0
    prec = tp/(tp+fp) if tp+fp > 0 else 0
    rec  = tp/(tp+fn) if tp+fn > 0 else 0
    f1   = 2.0*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
    print "%d %d %d %d %f %f %f"%(tp,tn,fp,fn,prec,rec,f1)                                      

if __name__ == '__main__':
    # tag the test data
    most_likely_tag = train('data/ne_files/Reuters_NER_train.txt')
    print(most_likely_tag)
    tag_testdata(most_likely_tag, 'data/ne_files/Reuters_NER_test.txt')
    # validation
    most_likely_tag = train('data/ne_files/train_split.txt')
    validate(most_likely_tag, 'data/ne_files/validation_split.txt')
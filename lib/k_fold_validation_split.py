K = 10 # value for K in K-fold validation

if __name__ == '__main__':
    sentence_count = 0
    with open('data/pos_files/train.pos','r') as f:
        data = f.readlines()
        for line in data:
            if line == '<s> <s>\n':
                sentence_count += 1
        validation_size = sentence_count / K
        training_size = sentence_count - validation_size
        for setnumber in range(K):
            with open('data/pos_files/validation_split%d.pos' % setnumber,'w') as fvalidate:
                with open('data/pos_files/train_split%d.pos' % setnumber,'w') as ftrain:
                    sentence_at = 0
                    for line in data:
                        if line == '<s> <s>\n':
                            sentence_at += 1
                        if setnumber*validation_size < sentence_at and sentence_at <= (setnumber+1)*validation_size:
                            fvalidate.write(line)
                        else:
                            ftrain.write(line)
    print("done")
                        
                
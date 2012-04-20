import argparse
from Parser import *
import cPickle as pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates kaggle submission files.')
    parser.add_argument('-d', type=str, metavar='DIR', dest='outdir', default='data/features/',
                       help='Where to load pickled data')
    parser.add_argument('-i', type=argparse.FileType('r'), metavar='FILE', dest='infile',
                       default=None, required=True,
                       help='Where to get output file from svm_hmm_classify')
    parser.add_argument('-t', type=argparse.FileType('r'), metavar='FILE', dest='testfile',
                       default=None, required=True,
                       help='Where to get test file with words')
    parser.add_argument('-o', type=argparse.FileType('w'), metavar='FILE', dest='outfile',
                       default=None, required=True,
                       help='Where to save kaggle output')
    
    args = parser.parse_args()
    print(args)
    
    with open(args.outdir+"pos.dat",'r') as f:
        pos = pickle.load(f)
    # POS_to_idx = dict( zip(pos,range(1,1+len(pos))) )
    idx_to_POS = dict( zip(range(1,1+len(pos)),pos) )
    
    labels = [ idx_to_POS[int(l.strip())] for l in args.infile ]
    args.infile.close()
    
    test_data = parse_opened_test_file(args.testfile)
    output_file = args.outfile
    idx = 0
    for sequence in test_data:
        output_file.write("<s> <s>\n")
        words = sequence[1:]
        for word in words:
            output_file.write(labels[idx]+" "+word+"\n")
            idx += 1
    assert idx == len(labels)
    output_file.close()
    args.testfile.close()
import argparse,os
from pprint import pprint
from Parser import *
from Features_impl import *

# List of types of features to use
FeatureSet = [  CapitalizedFeature,
                WordFeature,
                PrefixSuffixFeature,
                WordLengthFeature,
                LetterFrequencyFeature,
                SentenceLengthFeature
                # PunctuationFeature,
                # NumberFeature,
                ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates training/test files.')
    parser.add_argument('-n', metavar='N_TOP', type=int, dest='ntop', default=1000, 
                       help='number of top prefix/suffix to use')
    parser.add_argument('-w', metavar='WINDOW_SIZE', type=int, dest='windowsize', default=3, 
                       help='Size of feature window')
    parser.add_argument('-g', dest='generate', action='store_true',
                       help='Just generate feature set(s) and not actual features for train/test data')
    parser.add_argument('-d', type=str, metavar='DIR', dest='outdir', default='data/features/',
                       help='Where to save/load pickled data')
    parser.add_argument('--train', type=argparse.FileType('r'), metavar='FILE', dest='trainfile',
                       # default='data/pos_files/train.pos',
                       default=None,
                       help='Where to get training data')
    parser.add_argument('--test', type=argparse.FileType('r'), metavar='FILE', dest='testfile',
                       # default='data/pos_files/test-obs.pos',
                       default=None,
                       help='Where to get test data')
    
    args = parser.parse_args()
    pprint(args)
    
    force_generate = False
    try:
        with open(args.outdir+"pos.dat",'r') as f:
            pos = pickle.load(f)
        POS_to_idx = dict( zip(pos,range(1,1+len(pos))) )
    except:
        force_generate = True
        
    if args.generate or force_generate:
        data = parse_training_file()
        datalen = float(len(data))
        # fix_extractor = PrefixSuffixExtractor()
        # fix_extractor.train(data)
        feat = PrefixSuffixExtractor().train(data).get(ntop=args.ntop)
        with open(args.outdir+"prefix_suffix.dat","w") as f:
            pickle.dump(feat,f)
        feat = WordExtractor().train(data).get()
        with open(args.outdir+"words.dat","w") as f:
            pickle.dump(feat,f)
        feat = POSExtractor().train(data).get()
        with open(args.outdir+"pos.dat","w") as f:
            pickle.dump(feat,f)
    if args.generate == False:
        window = range( -(args.windowsize/2), 1+args.windowsize/2 )
        print "Window:"," ".join(str(window))
        fv = FeatureVectorizer(window=window,
                features=[f() for f in FeatureSet])
        # fv = FeatureVectorizer(features=[WordFeature()])
        print "Total number of features:",fv.len()
        # exit(0)
        if args.trainfile != None:
            training_data = parse_opened_training_file(args.trainfile)
            with open(args.trainfile.name + ".features",'w') as output_file:
                for qid,sequence in enumerate(training_data):
                    qid = qid+1
                    words = [w for p,w in sequence[1:]]
                    tags =  [POS_to_idx[p] for p,w in sequence[1:]]
                    for position in range(0,len(words)):
                        g = fv.transform(words,position)
                        tag = tags[position]
                        line = "%d qid:%d "%(tag,qid)
                        output_file.write(line)
                        line = " ".join( ["%d:%g"%(i,v) for i,v in g] )
                        output_file.write(line)
                        line = " # %s\n"%words[position]
                        output_file.write(line)
        if args.testfile != None:
            training_data = parse_opened_test_file(args.testfile)
            with open(args.testfile.name + ".features",'w') as output_file:
                tag = 1
                for qid,sequence in enumerate(training_data):
                    qid = qid+1
                    words = sequence[1:]
                    for position in range(0,len(words)):
                        g = fv.transform(words,position)
                        line = "%d qid:%d "%(tag,qid)
                        output_file.write(line)
                        line = " ".join( ["%d:%g"%(i,v) for i,v in g] )
                        output_file.write(line)
                        line = " # %s\n"%words[position]
                        output_file.write(line)
    
    
    
    
    
    
    
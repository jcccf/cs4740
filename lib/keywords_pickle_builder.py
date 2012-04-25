import os, cPickle as pickle

#to use, download the file linked http://cogcomp.cs.illinois.edu/Data/QA/QC/QC.tar
#and extract into data/train/qc/
if __name__ == '__main__':
    keywords = dict()
    for dirname, dirnames, filenames in os.walk('data/train/qc/publish/lists'):
        for filename in filenames:
            with open(os.path.join(dirname, filename), 'r') as f:
                for line in f:
                    line = line.replace('\n','')
                    keywords[line] = filename
    # Write to a pickle
    with open('data/train/qc/keywords2pickle', 'wb') as fout:
        pickle.dump(keywords, fout)

    
                    
import os, cPickle as pickle
from pprint import pprint

def load_keywords(try_cache=True, dir='data/train/qc/publish/lists'):
    if try_cache:
        try:
            with open('data/train/qc/keywords2pickle', 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    keywords = dict()
    for dirname, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if "." in filename:
                continue
            with open(os.path.join(dirname, filename), 'r') as f:
                for line in f:
                    line = line.replace('\n','')
                    keywords[line] = filename
    save_keywords(keywords=keywords)
    return keywords

def save_keywords(keywords):
    # Write to a pickle
    with open('data/train/qc/keywords2pickle', 'wb') as fout:
        pickle.dump(keywords, fout)
    # pprint(keywords)

#to use, download the file linked http://cogcomp.cs.illinois.edu/Data/QA/QC/QC.tar
#and extract into data/train/qc/
if __name__ == '__main__':
    load_keywords(try_cache=False)

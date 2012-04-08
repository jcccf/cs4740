import subprocess as sp
import shlex,sys,os

# Make cache directories if they don't exist
try:
  os.makedirs('data/output/')
except:
  pass

# Set to False if features have been generated already
GENERATE_FEATURE_FILES = False
# GENERATE_FEATURE_FILES = True

# Range of values for SVM parameter to prevent overfitting
# C_range = [0.1,1,10,100,1000,10000,100000]
C_range = [10000]

# File names for training and testing.  Can be pointed to
# development and validation files instead
train_file = 'data/pos_files/train.pos'
test_file = 'data/pos_files/test-obs.pos'

# svm_hmm is platform dependent, so we check if the platform
# is Windows or unix-like
is_win = os.name is "nt"

if __name__ == "__main__":
    if GENERATE_FEATURE_FILES:
        cmd = 'python Features.py --train %s --test %s'%(train_file, test_file)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
        
    # Try all parameters
    for C in C_range:
        # Learn SVM model
        cmd = '%s -c %g %s.features data/output/model_%g.txt'
        exe = 'svm_hmm_learn.exe' if is_win else './svm_hmm_learn'
        cmd = cmd%( exe, C, train_file, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
        
        # Predict sequence for test set
        cmd = '%s %s.features data/output/model_%g.txt data/output/out_%g.txt'
        exe = 'svm_hmm_classify.exe' if is_win else './svm_hmm_classify'
        cmd = cmd%( exe, test_file, C, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
        
        # Convert SVM's predictions (nominal) to actual parts-of-speech tags
        cmd = 'python generate_kaggle.py -t %s -i data/output/out_%g.txt -o data/output/kaggle_%g.txt'
        cmd = cmd%( test_file, C, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()


import subprocess as sp
import shlex,sys,os

# Make cache directories if they don't exist
try:
  os.makedirs('data/output/')
except:
  pass
  
GENERATE_FEATURE_FILES = False
C_range = [10000]
is_win = "win" in sys.platform
if __name__ == "__main__":
    if GENERATE_FEATURE_FILES:
        cmd = 'python Features.py --train data/pos_files/train.pos --test data/pos_files/test-obs.pos'
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
    # Train svm-hmm models and generate output on test data
    for C in C_range:
        cmd = '%s -c %g data/pos_files/train.pos.features data/output/model_%g.txt'
        exe = 'svm_hmm_learn.exe' if is_win else './svm_hmm_learn'
        cmd = cmd%( exe, C, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
        
        cmd = '%s data/pos_files/test-obs.pos.features data/output/model_%g.txt data/output/out_%g.txt'
        exe = 'svm_hmm_classify.exe' if is_win else './svm_hmm_classify'
        cmd = cmd%( exe, C, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()
        
        cmd = 'python generate_kaggle.py -t data/pos_files/test-obs.pos -i data/output/out_%g.txt -o data/output/kaggle_%g.txt'
        cmd = cmd%( C, C )
        cmd = cmd.replace('/',os.sep)
        print cmd
        p = sp.Popen(cmd,shell=True)
        out,err = p.communicate()

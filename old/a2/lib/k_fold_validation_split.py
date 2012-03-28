import re
from math import floor

K = 8 # value for K in K-fold validation

line_matcher = re.compile(r'([\w]+)\.([\w]+) ([0-9 ]+)@[ ]*(.*)@([\w]+)@(.*)')
with open('data/wsd-data/train.data','r') as f:
    data = f.readlines()
# get counts for each word so we know how much to split by
countsdict = dict()
for line in data:
    match_obj = line_matcher.match(line)
    if match_obj:
        word, pos, senses, context_before, target, context_after = match_obj.groups()
        if word in countsdict:
            countsdict[word] += 1
        else:
            countsdict[word] = 1
    else:
        raise Exception("Example Regex Failed to Match on\n%s" % l)
# divide each of the counts in countsdict by k to get number 
# for each word to put in validation set
for (key,value) in countsdict.iteritems():
    countsdict[key] = floor(value / K)
# split the files for k-fold cross validation
currentword = ''
wordindex = 0
for setnumber in range(K):
    with open('data/wsd-data/train_split%d.data' % setnumber,'w') as ftrain:
        with open('data/wsd-data/valiation_split%d.data' % setnumber,'w') as fvalidate:
            for line in data:
                match_obj = line_matcher.match(line)
                if match_obj:
                    word, pos, senses, context_before, target, context_after = match_obj.groups()
                else:
                    raise Exception("Example Regex Failed to Match on\n%s" % l)
                if (currentword != word):
                    # reached a different word
                    currentword = word
                    wordindex = 0
                else:
                    wordindex += 1
                if wordindex >= setnumber*countsdict[word] and wordindex < (setnumber+1)*countsdict[word]:
                    fvalidate.write(line)
                else:
                    ftrain.write(line)
    print("set %d done" % setnumber)
print("done")
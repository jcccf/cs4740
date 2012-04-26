#!/bin/sh
scriptdir=`dirname $0`

java -mx700m -cp "$scriptdir/stanford-ner.jar:" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier $scriptdir/classifiers/english.muc.7class.distsim.crf.ser.gz -textFile $1

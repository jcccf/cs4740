CS 4740 Report - Word Sense Disambiguation
Justin Cheng (jc882) , Marcus Lim (mkl65), Yi heng Lee (yl478)

Requirements
- Python 2.7+
- nltk and sklearn packages (and their dependencies)
- libxml (for the xml parser)
- Stanford Parser (http://nlp.stanford.edu/software/lex-parser.shtml) if syntactic dependencies features are used

Use
0. Unzip data files to the "data" folder.
1. Use split_validation.py to generate training and validation sets. (there is also a k-fold validation splitter, but doing k-fold validation took too long)
2. Use scikit_classifier.py to compute scores. (use "python lib/scikit_classifier.py --help" for options
3. For syntactic dependencies, the training and validation sets must be prepared for the Stanford Parser by running Syntactic_features.py, changing the variable filename to the file you want to parse, to generate files such as "train_split.data.sout". The lexparser.sh file of the Stanford Parser is then edited to run with the options -writeOutputFiles -outputFormat "oneline,typedDependencies". The Stanford Parser is run from the command line using lexparser.sh on each of the .sout files to generated parsed .sout.stp files. Place the .sout.stp files in the data/wsd-data so that that our code can parse the dependencies from them.
4. combinatorial_search.py can be used to automatically run the classifier over a range of feature parameters, the parameters can be changed by editing the for statements. 

Notes
- Baselinemostfrequentsense.py computes our baseline (predicting the most frequent word sense).
- the sample input to/output from stanford parser.txt files contain examples of the contents of files described in 3.
- lib/console_output.txt is an example of text printed to the console when the classifier is run. Since the parsers for the POS, co-occurence, Lesk, N-gram features directly pass the data to the classifier, they do not generate any output files.

Our QA System

How to use
==========
1. Place the train and test files in /data, within the /lib folder.
2. Modify the line starting with "DIR" in Loader.py, depending on whether you want to evaluate the train set or the test set.
3. Run Parser.py to prepare the documents in a nicer format. These include the documents (-d) and the questions (-q).
4. Read the top of Parser.py to find out what kind of output Parser.py generates
5. Run CoreNLP and generate CoreNLP parsed documents (see separate section below).
6. Run Pipeline.py to generate answers to the range of questions specified at the command line.

CoreNLP
=======
1. Download CoreNLP from here and untar to /corenlp (directly, not in its own directory).
I.e. corenlp.py will reside beside a lot of random jar files in the corenlp directory.
http://nlp.stanford.edu/software/stanford-corenlp-2012-04-09.tgz

2. There might be several dependencies you need to install, including
"simplejson", "pexpect" and "unidecode".

3. Start the CoreNLP server with python corenlp.py (run from /corenlp).

4. Parse some documents using "python Parser.py -d". You only need to do this once, as the parsed documents are saved.

5. To load and parse documents, use CoreNLPLoader in CoreNLPLoader.py. Also run CoreNLPLoader.py with the option -q to parse questions.

6. To see what kind of data CoreNLP obtains, see the class CoreNLPFeatures in CoreNLPParser.py.

Additional Notes
=======
- To see what kind of options are available for Parser.py, CoreNLPLoader.py, and Pipeline.py, run each file with the option "--help".

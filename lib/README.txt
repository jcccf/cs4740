A Magic QA System

How to use
==========
1. Place the train files in /data, within the /lib folder
2. Run Parser.py to prepare the documents in a nicer format
3. Read the top of Parser.py to find out what kind of output Parser.py generates

CoreNLP
=======
1. Download CoreNLP from here and untar to /corenlp (directly, not in its own directory).
I.e. corenlp.py will reside beside a lot of random jar files in the corenlp directory.
http://nlp.stanford.edu/software/stanford-corenlp-2012-04-09.tgz
   a. Install the following packages with pip:
        unidecode, progressbar

2. There might be several dependencies you need to install, including
"simplejson", "pexpect" and "unidecode".

3. Start the CoreNLP server with python corenlp.py (run from /corenlp).

4. Parse some documents using parse_docs() in Parse.py (1st-pass parse).

5. To load and parse documents, use CoreNLPLoader in CoreNLPLoader.py.

6. To see what kind of data CoreNLP obtains, see the class CoreNLPFeatures in CoreNLPParser.py.

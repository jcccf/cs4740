CS 4740 Report - Word Sense Disambiguation
Justin Cheng (jc882) , Marcus Lim (mkl65), Yi heng Lee (yl478)

Requirements
- Python 2.7+
- nltk and sklearn packages (and their dependencies)
- libxml (for the xml parser)

Use
1. Use split_validation.py to generate training and validation sets.
2. Use scikit_classifier.py to compute scores. (use "python lib/scikit_classifier.py --help" for options)

Notes
- Baselinemostfrequentsense.py computes our baseline (predicting the most frequent word sense).

tl;dr version
-------------
Extract datasets to "data" folder, and from the root folder (not lib), run "python lib/main.py"

Long version
------------
You require Python 2.7+ and nltk in order to run our code. Unzip the package to a directory of your liking. There should be a folder called “lib”, which contains all the code that we used in this assignment. All data should be placed in a separate directory called “data” (at the same level as the “lib” folder), and all datasets decompressed to that folder (1 dataset per folder). In addition, create a folder called “output” in the “data” folder for the data to be generated.
From the main directory (not “lib”), run “python lib/main.py”. You will be asked what you want to run. Type in 1 to generate random sentences, 2 to compute perplexities, 3 to predict authors in the Enron dataset, and 4 to generate accuracies for author prediction.
You can further specify which type of n-gram to run in the first few lines of main.py. By default, all possible permutations are computed, so that the script may take up to several hours to complete.
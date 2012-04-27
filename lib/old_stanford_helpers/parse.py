# Deprecated, use CoreNLP instead
# Generate part-of-speech tags and identify named entities
def generate_pos_ne():
  print "Generating POS Tags/NEs"
  try:
    os.makedirs('data/train/parsed_docs_posne')
  except:
    pass
  for filename in glob.glob("data/train/parsed_docs/top_docs.*"):
    print filename
    pdocs = []
    with open(filename, 'rb') as f:
      docs = pickle.load(f)
      for doc in docs:
        pdoc = {}
        for k in doc.keys(): # Generate POS tags and NEs for each key in the document
          if doc[k] is not None and doc[k] is not "docno":
            # Split into sentences, then words, POS tag words, then chunk tagged words
            sentences = nltk.sent_tokenize(doc[k])
            allsent, allpos, allne, allptree = [], [], [], []
            for s in sentences:
              words = nltk.word_tokenize(s)
              postags = nltk.pos_tag(words)
              nes = nltk.ne_chunk(postags)
              tree = Chunker.chunker.parse(postags)
              allsent.append(words)
              allpos.append(postags)
              allne.append(nes)
              allptree.append(tree)
            pdoc[k] = { "sentences": allsent, "pos": allpos, "ne": allne, "parse_tree": allptree }
          else:
            pdoc[k] = None
        pdocs.append(pdoc)
    # Write to a pickle
    with open('data/train/parsed_docs_posne/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(pdocs, f)


# /parsed_docs_trees
#   List of { "text" : { 
#       "sentences": List of List of Words,
#       "pos_tags": List of List of POS tuples, 
#       "parse_trees": List of Parse Trees, 
#       "np_chunks": List of NP Chunks, 
#       "nes": List of Named Entities (3 types), 
#       "nes7": List of Named Entities (7 types)
#     },
#     "leadpara" : { same as before },
#     and so on
#   }

# Deprecated, use CoreNLP instead
def generate_parse_trees():
  print "Generating Parse Trees using the Stanford Parser..."
  try:
    os.makedirs('data/train/parsed_docs_trees')
  except:
    pass
  for filename in glob.glob("data/train/parsed_docs/top_docs.*"):
    if int(filename.rsplit(".", 1)[1]) <= 210:
      continue
    print filename
    pdocs = []
    with open(filename, 'rb') as f:
      docs = pickle.load(f)
      for docindex, doc in enumerate(docs):
        print doc['docno']
        pdoc = {}
        with open('temp.txt', 'w') as f:
          for k in ['headline', 'leadpara', 'text']:
            if doc[k] is not None:
              f.write(doc[k]+"\n")
        sentences, pos_tags, parse_trees, np_chunks, nes, nes7 = [], [], [], [], [], []
        # Get NEs (3 types)
        lines = subprocess.check_output(["../../../tools/stanford-ner-2012-04-07/ner.sh", "temp.txt"])
        lines = lines.strip()
        for i, l in enumerate(lines.split("\n")):
          ne = [tuple(w.split("/")) for w in l.split(" ") if len(w) > 0]
          if len(ne) > 100: # attempt to split
            temp_ne = []
            for n in ne:
              if (n[0] == ';' or n[0] == '.') and len(temp_ne) > 0:
                temp_ne.append(n)
                nes.append(temp_ne)
                temp_ne = []
              else:
                temp_ne.append(n)
            if len(temp_ne) > 0:
              nes.append(temp_ne)
          else:
            nes.append(ne)
        # Get NEs (7 types)
        lines = subprocess.check_output(["../../../tools/stanford-ner-2012-04-07/nes.sh", "temp.txt"])
        lines = lines.strip()
        for i, l in enumerate(lines.split("\n")):
          ne = [tuple(w.split("/")) for w in l.split(" ") if len(w) > 0]
          if len(ne) > 100: # attempt to split
            temp_ne = []
            for n in ne:
              if (n[0] == ';' or n[0] == '.') and len(temp_ne) > 0:
                temp_ne.append(n)
                nes7.append(temp_ne)
                temp_ne = []
              else:
                temp_ne.append(n)
            if len(temp_ne) > 0:
              nes7.append(temp_ne)
          else:
            nes7.append(ne)
        # Write tokenized strings to file again so that the NE and Parse Tree tokenization match up exactly
        with open('temp.txt', 'w') as f:
          for ne in nes:
            f.write(" ".join([w[0] for w in ne])+"\n")
        lines = subprocess.check_output(["../../../tools/stanford-parser-2012-03-09/parsetree.sh", "temp.txt"])
        lines = lines.strip()
        # Get POS Tags and Parse Trees
        for i, l in enumerate(lines.split("\n")):
          if i % 3 == 0: # POS Tags
            pos = [tuple(w.split("/")) for w in l.split(" ")]
            pos_tags.append(pos)
            sentences.append([w[0] for w in pos])
            chunks = Chunker.chunker.parse(pos)
            np_chunks.append(chunks)
          elif i % 3 == 2: # Parse Tree
            tree = nltk.tree.Tree.parse(l)
            parse_trees.append(tree)
        # Make sure everything matches up
        try:
          assert len(pos_tags) == len(parse_trees) == len(nes) == len(nes7)
        except:
          print doc['docno'], len(pos_tags), len(nes)
          for i, pos_tag in enumerate(pos_tags):
            print i, " ".join([w[0] for w in pos_tag])
          for i, ne in enumerate(nes):
            print i, " ".join([w[0] for w in ne])
          raise Exception
        # Okay, add to list of parsed documents
        pdocs.append({ "docno": doc['docno'], "sentences": sentences, "pos_tags": pos_tags, "parse_trees": parse_trees, "np_chunks": np_chunks, "nes": nes, "nes7": nes7 })
    with open('data/train/parsed_docs_trees/%s' % os.path.basename(filename), 'wb') as f:
      pickle.dump(pdocs, f)

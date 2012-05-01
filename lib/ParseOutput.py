from pprint import pprint

filename = "wn_def_keywords_desc_qns.txt"

if __name__ == "__main__":
  answers = {}
  with open(filename,"r") as f:
    for qno in range(201,400):
      anss = []
      for idx in range(5):
        line = f.readline()
        anss.append( line.split()[2:] )
      answers[qno] = anss
  pprint(answers[234])
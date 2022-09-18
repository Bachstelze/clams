import os

languages = ["de","en","ru","fr","he"]
test_cases = [
"long_vp_coord",
"obj_rel_across_anim",
"obj_rel_within_anim",
"prep_anim",
"simple_agrmt",
"subj_rel",
"vp_coord"
]

for test_case in test_cases:
  os.mkdir(test_case)
  for lang in languages:
    directory = './clams/'+lang+'_evalset/'
    with open(test_case+"/"+lang+'_forbert.tsv', 'w') as outfile:
      with open(directory+test_case+".txt", 'r') as infile:
          g = ""
          ug = ""
          for line in infile:
              line_tuple = line.strip().split('\t')
              if line_tuple[0] == "True":
                  is_grammatical = True
              else:
                  is_grammatical = False
              sentence = line_tuple[1]

              if is_grammatical:
                  g = sentence
              else:
                  ug = sentence
                  outfile.write(test_case+'\t'+lang+'\t'+g+'\t'+ug+'\n')

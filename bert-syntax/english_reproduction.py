# evaluate in batches with the huggingface pipeline

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import sys
import csv
from collections import Counter

# Use enlish model
model_name = 'bert-base-uncased' # 'bert-base-multilingual-cased'
en_tokenizer = AutoTokenizer.from_pretrained(model_name)
en_model = AutoModelForMaskedLM.from_pretrained(model_name, is_decoder=False)
#model.eval()

fill_pipeline = pipeline("fill-mask",
               model=en_model,
               tokenizer=en_tokenizer,
               device=0)

languages = ["en"]

def read_test(lang, directory):
  cc = Counter()
  # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
  out = []
  for line in open(directory + "/" + lang+"_forbert.tsv"):
      case = line.strip().split("\t")
      cc[case[0]]+=1
      g,ug = case[-2],case[-1]
      g = g.split()
      ug = ug.split()
      assert(len(g)==len(ug)),(g,ug)
      diffs = [i for i,pair in enumerate(zip(g,ug)) if pair[0]!=pair[1]]
      if (len(diffs)!=1):
          #print(diffs)
          #print(g,ug)
          continue
      assert(len(diffs)==1),diffs
      gv=g[diffs[0]]   # good
      ugv=ug[diffs[0]] # bad
      g[diffs[0]]="***mask***"
      g.append(".")
      out.append((case[0],case[1]," ".join(g),gv,ugv))
  return out

for lang in languages:
    result_file=open(lang+'_result_27_09.txt', 'a')
    test_list = read_test(lang, ".")
    target_tokens = set()
    for line in test_list:
        target_tokens.add(line[-1])
        target_tokens.add(line[-2])

    datasets = {} # the target token is the key
    dataset_mapping = {}  # the target token is the key
    dataset_cases = {}
    target_scores = {}  # the target token is the key
    for target_token in list(target_tokens):
        datasets[target_token] = []
        dataset_mapping[target_token] = []
        dataset_cases[target_token] = []
        target_scores[target_token] = []

    line_result = {} # the line is the key
    for line_number, line in enumerate(test_list):
        case,tp,s,word1,word2 = line
        pre,target,post=s.split('***')
        combined_sentence = pre + en_tokenizer.mask_token + post

        datasets[word1].append(combined_sentence)
        dataset_mapping[word1].append(line_number)
        dataset_cases[word1].append(case)

        datasets[word2].append(combined_sentence)
        dataset_mapping[word2].append(line_number)
        dataset_cases[word2].append(case)

        line_result[line_number] = [[word1, word2], {word1:[], word2:[]}]

    if model_name == 'bert-base-multilingual-cased':
        remove_tokens = ["swims", "swim", "admires", "admire", "laugh", "laughs",
        "hates", "hate", "enjoys", "enjoy", "like", "likes", "smile", "smiles"]
    else:
        remove_tokens = ["swims", "swim","admires", "admire"]
    for remove_token in remove_tokens:
        target_tokens.remove(remove_token)
    print(target_tokens)

    for token in target_tokens:
        fill_pipeline( en_tokenizer.mask_token, targets=[token])



    print(len(target_tokens))
    #print(datasets["like"])
    #print(dataset_mapping["like"])

    # calculate the scores for each target token
    batch_size = 8
    for target_token in target_tokens:
        print(target_token)
        for line in datasets[target_token]:
            filled_line = fill_pipeline(line, targets=[target_token], batch_size=batch_size)
            target_scores[target_token].append(filled_line[0]["score"])
    #print(target_scores["like"])

    # summarize to scores for each line
    for target_token in target_tokens:
        for array_id, test_number in enumerate(dataset_mapping[target_token]):
            case = dataset_cases[target_token][array_id]
            score = target_scores[target_token][array_id]
            line_result[test_number][1][target_token] = [case, score]

    # print the results
    for line_number in range(len(test_list)):
        print(line_number)
        #load values from the loaded data
        case, lang, sentence, w1, w2 = test_list[line_number]

        # only process if we calculated the scores
        if w1 in target_tokens:
            word1 = line_result[line_number][0][0]
            word2 = line_result[line_number][0][1]
            case1, score1 = line_result[line_number][1][word1]
            case2, score2 = line_result[line_number][1][word2]
            correctness = score1 > score2

            #verify valid data structure
            assert case == case1 and case == case2
            assert w1 == word1
            assert w2 == word2
            print(correctness, case, lang, w1, str(score1), w2, str(score2), sentence,file=result_file)
        #print(str(correctness) + " " + case1 + " en " + word1 + " " + str(score1) + " " + word2 + " " + str(score2))
        #print(gp>bp,case,tp,g,b,s,file=result_file)

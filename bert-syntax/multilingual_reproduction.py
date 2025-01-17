# evaluate in batches with the huggingface pipeline

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import sys
import csv
from collections import Counter

def get_pipeline_output(fill_pipeline, model_name, input, target_token, batch_size):
    if model_name.endswith('xlm-roberta-base'):
        target_token = '▁' + target_token
    return fill_pipeline(input, targets=[target_token], batch_size=batch_size)

def read_test(lang, directory, file_name):
  cc = Counter()
  # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
  out = []
  for line in open(directory + "/" + lang+file_name):
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

def get_target_tokens(test_list):
    target_tokens = set()
    for line in test_list:
        target_tokens.add(line[-1])
        target_tokens.add(line[-2])
    return target_tokens

def prepare_dataset_dictionaries(tokenizer, test_list, target_tokens):
    """
    preparation for a map reduce like algorithm
    the target token is the key of the dictionaries
    """
    datasets = {}
    dataset_mapping = {}
    dataset_cases = {}
    target_scores = {}
    for target_token in list(target_tokens):
        datasets[target_token] = []
        dataset_mapping[target_token] = []
        dataset_cases[target_token] = []
        target_scores[target_token] = []

    line_result = {} # the line is the key
    for line_number, line in enumerate(test_list):
        case,tp,s,word1,word2 = line
        pre,target,post=s.split('***')
        combined_sentence = pre + tokenizer.mask_token + post

        datasets[word1].append(combined_sentence)
        dataset_mapping[word1].append(line_number)
        dataset_cases[word1].append(case)

        datasets[word2].append(combined_sentence)
        dataset_mapping[word2].append(line_number)
        dataset_cases[word2].append(case)

        line_result[line_number] = [[word1, word2], {word1:[], word2:[]}]

    return datasets, dataset_mapping, dataset_cases, target_scores, line_result

def get_out_of_vocabulary_tokens(fill_pipeline, tokenizer, model_name, target_tokens):
    out_of_vocabulary_tokens = []
    input = tokenizer.mask_token

    for token in target_tokens:
        token_str = get_pipeline_output(fill_pipeline, model_name, input, token, 1)[0]['token_str']
        if token_str != token:
            out_of_vocabulary_tokens.append(token)

    print("out_of_vocabulary_tokens:")
    print(out_of_vocabulary_tokens)

    for remove_token in out_of_vocabulary_tokens:
        target_tokens.remove(remove_token)

    print("full token list:")
    print(target_tokens)

    # there should be no more prints
    for token in target_tokens:
        get_pipeline_output(fill_pipeline, model_name, input, token, 1)

    return out_of_vocabulary_tokens

def map_reduce_scores(fill_pipeline, model_name, batch_size, target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result):
    for target_token in target_tokens:
        print(target_token)
        for line in datasets[target_token]:
            filled_line = get_pipeline_output(fill_pipeline, model_name, line, target_token, batch_size)
            target_scores[target_token].append(filled_line[0]["score"])

    # summarize to scores for each line
    for target_token in target_tokens:
        for array_id, test_number in enumerate(dataset_mapping[target_token]):
            case = dataset_cases[target_token][array_id]
            score = target_scores[target_token][array_id]
            line_result[test_number][1][target_token] = [case, score]

def print_results(test_list, target_tokens, line_result, result_file):
    for line_number in range(len(test_list)):
        #load values from the loaded data
        case, lang, sentence, w1, w2 = test_list[line_number]

        # only process if we calculated the scores
        if w1 in target_tokens and w2 in target_tokens:
            word1 = line_result[line_number][0][0]
            word2 = line_result[line_number][0][1]
            case1, score1 = line_result[line_number][1][word1]
            case2, score2 = line_result[line_number][1][word2]
            correctness = score1 > score2

            # verify valid data structure
            assert case == case1 and case == case2
            assert w1 == word1
            assert w2 == word2
            # print the results to the defined file
            print(correctness, case, lang, w1, str(score1), w2, str(score2), sentence,file=result_file)

if __name__ == "__main__":
    #model_name = 'xlm-roberta-base' # multilingual roberta
    model_name = 'bert-base-multilingual-cased' # multilingual
    #model_name = 'bert-base-uncased' # monolingual

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, is_decoder=False)
    #model.eval()

    fill_pipeline = pipeline("fill-mask",
                   model=model,
                   tokenizer=tokenizer,
                   device=0)

    languages = ["de","en","ru","fr","he"]

    for lang in languages:
        result_file=open(lang+'_result_xlm_04_10.txt', 'a')
        test_list = read_test(lang, ".", "_forbert_subset.tsv")
        target_tokens = get_target_tokens(test_list)
        datasets, dataset_mapping, dataset_cases, target_scores, line_result = prepare_dataset_dictionaries(tokenizer, test_list, target_tokens)

        out_of_vocabulary_tokens = get_out_of_vocabulary_tokens(fill_pipeline, tokenizer, model_name, target_tokens)
        print(len(target_tokens))

        # calculate the scores for each target token
        batch_size = 8
        map_reduce_scores(fill_pipeline, model_name, batch_size, target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result)

        # print the results
        print_results(test_list, target_tokens, line_result, result_file)

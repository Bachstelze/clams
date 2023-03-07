from sys import argv
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import sys
import csv
from collections import Counter
import math
from collections import *

script, modelname, batch_size = argv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t5_tokenizer = T5Tokenizer.from_pretrained(modelname)
t5_config = T5Config.from_pretrained(modelname)
#print(t5_config)
t5_model = T5ForConditionalGeneration.from_pretrained(modelname, config=t5_config).to(DEVICE)
#t5_mlm.eval()
t5_tokenizer.mask_token = "<extra_id_0>"

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

def chunks(full_list, max_number):
  n = math.ceil(len(full_list)/int(max_number))
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(full_list), n):
    yield full_list[i:i + n]
    
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
    
def get_target_score(sentences, target, beam_searches=2):
  """
  The inference function for the mask fill-in
  Parameters:
    text(String): The input text with <extra_id_0> as mask
    target(String): The target word
    beam_searches(int): The number of beam searches and results
  """

  results = []

  # encode the target word
  encoded_target_ids = t5_tokenizer(target, add_special_tokens=False).input_ids
  #print(target, encoded_target_ids)
  token_number = len(encoded_target_ids)

  # chunk the sentences into the maximum number of processed sentences at once
  sentence_list_chunks = list(chunks(sentences, batch_size))

  for sentence_list in sentence_list_chunks:
    # encode the input text
    encoded_text = t5_tokenizer(sentence_list, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    input_text_ids = encoded_text['input_ids'].to(DEVICE)
    # generate the outputs with constrained search and the target as constrains
    outputs = t5_model.generate(input_ids=input_text_ids,
                            force_words_ids=[encoded_target_ids],
                            num_beams=beam_searches,
                            num_return_sequences=beam_searches,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_length=token_number+1) # plus one token for the <pad>
    
    #print(outputs)

    # filter each output and save it
    for output_number, output in enumerate(outputs["sequences"]):
      _txt = t5_tokenizer.decode(output[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
      #print(_txt)
      if _txt == target:
        # save the target score
        results.append([outputs["sequences_scores"][output_number].item()])
        # complete text
        #print(_result_prefix + _txt + _result_suffix)


  # return the aggregated result
  return results

def map_reduce_scores(target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result):
    for target_token in target_tokens:
        print(target_token)
        filled_line = get_target_score(datasets[target_token], target_token)
        target_scores[target_token].extend(filled_line)

    # summarize to scores for each line
    for target_token in target_tokens:
        for array_id, test_number in enumerate(dataset_mapping[target_token]):
            case = dataset_cases[target_token][array_id]
            score = target_scores[target_token][array_id][0]
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
            
def generate_table(result_file_name):
  by_model={}
  conditions=set()
  lines = open(result_file_name)
  results=defaultdict(Counter)
  skipped = set()
  for line in lines:
      if line.startswith("Better speed"): continue
      if line.startswith("skipping"):
          skipped.add(line.split()[1])
          next(lines)
          continue
      res,c1,c2,w1,w2,s = line.split(None, 5)
      c1 = c1.replace("inanim","anim")
      conditions.add(c1)
      results[c1][res]+=1

  print("skipped:",skipped)

  print("condition & base & large & count \\\\")
  for cond in conditions:
      rb = results[cond]
      #rl = by_model['large'][cond]
      sb = "%.2f" % (rb['True']/(rb['True']+rb['False']))
      #sl = "%.2f" % (rl['True']/(rl['True']+rl['False']))
      #print(" & ".join(map(str,[cond, sb, sl, sum(rb.values())])),"\\\\")
      print(" & ".join(map(str,[cond, sb, sum(rb.values())])),"\\\\")
            
languages = ["de","en","ru","fr","he"]

for lang in languages:
    result_file_name = lang+'_result_with_'+modelname.split("/")[-1]+'.txt'
    result_file=open(result_file_name, 'w')
    test_list = read_test(lang, ".", "_forbert_subset.tsv")
    target_tokens = get_target_tokens(test_list)
    print("target_tokens:")
    print(target_tokens)
    datasets, dataset_mapping, dataset_cases, target_scores, line_result = prepare_dataset_dictionaries(t5_tokenizer, test_list, target_tokens)
    # calculate the scores for each target token
    map_reduce_scores(target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result)

    # print the results into the result file
    print_results(test_list, target_tokens, line_result, result_file)
    
    # summarize the results
    generate_table(result_file_name)

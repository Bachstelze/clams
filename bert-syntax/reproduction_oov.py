from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from multilingual_reproduction import *

def get_single_word_score(sent,target_word,target_end,word_combination, scores):
  pre,target,post=sent.split('***')
  combined_sentence = pre + word_combination + multilingual_tokenizer.mask_token + post
  filled_sentence = fill_pipeline(combined_sentence, targets=[target_end])
  filled_token = filled_sentence[0]["token_str"]
  word_combination += filled_token
  scores.append(filled_sentence[0]["score"])

  if word_combination == target_word:
    return sum(scores)/len(scores)
  elif target_end.startswith(filled_token):
    # the predicted token is not the target word,
    # but the the filled token is part of the target word
    new_sent = pre + filled_token + '***mask***' + post
    return get_single_word_score(sent,target_word,target_end[len(filled_token):],word_combination,scores)

  # the predicted token is not the target word
  # and the filled taoken is not part of the target word.
  print("No score for " + target_word)
  return None

def get_oov_pipeline_output(fill_pipeline, model_name, input, target_token, batch_size):
    if model_name.endwith('xlm-roberta-base'):
        target_token = '▁' + target_token
    return fill_pipeline(input, targets=[target_token], batch_size=batch_size)

def map_reduce_oov_scores(fill_pipeline, model_name, batch_size, target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result):
    for target_token in target_tokens:
        print(target_token)
        scores = get_pipeline_output(fill_pipeline, model_name, datasets[target_token], target_token, batch_size)
        for score in scores:
            target_scores[target_token].append(score["score"])

    # summarize to scores for each line
    for target_token in target_tokens:
        for array_id, test_number in enumerate(dataset_mapping[target_token]):
            case = dataset_cases[target_token][array_id]
            score = target_scores[target_token][array_id]
            line_result[test_number][1][target_token] = [case, score]

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
        test_list = read_test(lang, ".", "_forbert_subset.tsv")
        target_tokens = get_target_tokens(test_list)
        datasets, dataset_mapping, dataset_cases, target_scores, line_result = prepare_dataset_dictionaries(tokenizer, test_list, target_tokens)

        out_of_vocabulary_tokens = get_out_of_vocabulary_tokens(fill_pipeline, tokenizer, model_name, target_tokens)
        print(len(target_tokens))

        # calculate the scores for each target token
        batch_size = 8
        map_reduce_oov_scores(fill_pipeline, model_name, batch_size, target_tokens, datasets, target_scores, dataset_mapping, dataset_cases, line_result)

        # print the results
        file_ending = '_result_05_10.txt'
        result_file=open(lang + file_ending, 'a')
        print_results(test_list, target_tokens, line_result, result_file)

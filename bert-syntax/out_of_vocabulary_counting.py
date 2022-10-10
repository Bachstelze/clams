from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from multilingual_reproduction import read_test, get_pipeline_output, get_target_tokens, get_out_of_vocabulary_tokens

multilingaul_model_names = ['xlm-roberta-base', 'bert-base-multilingual-cased']
english_model = 'bert-base-uncased'
languages = ["de","en","ru","fr","he"]
result = {}

def calculate_results(lang, model_name, input_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, is_decoder=False)
    fill_pipeline = pipeline("fill-mask",
                   model=model,
                   tokenizer=tokenizer,
                   device=0)

    test_list = read_test(lang, ".", input_file)
    target_tokens = get_target_tokens(test_list)
    full_length = len(target_tokens)

    out_of_vocabulary_tokens = get_out_of_vocabulary_tokens(fill_pipeline, tokenizer, model_name, target_tokens)
    oov_length = len(out_of_vocabulary_tokens)

    # print and save the results
    result[lang][model_name] = str(oov_length) + "/" + str(full_length)
    print(lang, model_name, full_length, oov_length)

#iterate over languages and model names
for lang in languages:
    result[lang] = {}
    for model_name in multilingaul_model_names:
        calculate_results(lang, model_name, "_forbert_subset.tsv")

calculate_results("en", english_model, "_forbert.tsv")

print("result overview:")
for lang in languages:
    if lang == "en":
        print(lang, english_model)
        print(result[lang][english_model])
    for model_name in multilingaul_model_names:
        print(lang, model_name)
        print(result[lang][model_name])

"""
terminal output:
result overview:

de xlm-roberta-base
9/20
de bert-base-multilingual-cased
12/20

en bert-base-uncased
2/33
en xlm-roberta-base
10/22
en bert-base-multilingual-cased
12/22

ru xlm-roberta-base
11/30
ru bert-base-multilingual-cased
23/30

fr xlm-roberta-base
14/24
fr bert-base-multilingual-cased
15/24

he xlm-roberta-base
22/36
he bert-base-multilingual-cased
32/36
"""

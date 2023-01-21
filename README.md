# Cross-Linguistic Assessment of Models on Syntax
The [Reproducing targeted syntactic evaluations of language models](https://raw.githubusercontent.com/Bachstelze/clams/master/Testing_the_reproducibility_of_syntactic_language_model_evaluations-2023.pdf) is based on CLAMS (*Cross-Linguistic Syntactic Evaluation of Word Prediction Models*, Mueller et al., ACL 2020).

Source and instructions required for replicating the [CLAMS paper](https://aclanthology.org/2020.acl-main.490.pdf).

In this repository, we provide the following:

- Our data sets for training and testing the LSTM language models
- Syntactic evaluation sets
	- English, French, German, Hebrew, Russian
- Attribute-varying grammars (AVGs)
- Code for replicating the CLAMS paper, including:
	- A modified form of [Rebecca Marvin's syntactic evaluation code](https://github.com/BeckyMarvin/LM_syneval) for testing on CLAMS evaluation sets
	- A modified form of [Yoav Goldberg's BERT-Syntax code](https://github.com/yoavg/bert-syntax), as well as code to prepare CLAMS evaluation sets for running in it

## Data Sets
- English: [train](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/train.txt) / [valid](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/valid.txt) / [test](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/test.txt) / [vocab](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt)
	- From Gulordava et al. (2018)
- French: [train](https://zenodo.org/record/3774712/files/fr_train.txt?download=1) / [valid](https://zenodo.org/record/3774712/files/fr_valid.txt?download=1) / [test](https://zenodo.org/record/3774712/files/fr_test.txt?download=1) / [vocab](https://zenodo.org/record/3774712/files/fr_vocab.txt?download=1)
- German: [train](https://zenodo.org/record/3774712/files/de_train.txt?download=1) / [valid](https://zenodo.org/record/3774712/files/de_valid.txt?download=1) / [test](https://zenodo.org/record/3774712/files/de_test.txt?download=1) / [vocab](https://zenodo.org/record/3774712/files/de_vocab.txt?download=1)
- Hebrew: [train](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/train.txt) / [valid](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/valid.txt) / [test](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/test.txt) / [vocab](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/vocab.txt)
	- From Gulordava et al. (2018)
- Russian: [train](https://zenodo.org/record/3774712/files/ru_train.txt?download=1) / [valid](https://zenodo.org/record/3774712/files/ru_valid.txt?download=1) / [test](https://zenodo.org/record/3774712/files/ru_test.txt?download=1) / [vocab](https://zenodo.org/record/3774712/files/ru_vocab.txt?download=1)

To replicate the multilingual corpora, simply concatenate the training, validation, and test corpora for each language. The multilingual vocabulary is the concatenation of each language's monolingual vocabulary.

## Attribute-Varying Grammars
These are used to generate syntactic evaluation sets by varying attributes. This generates sets of grammatical and ungrammatical examples in a controlled manner.

The behavior of this system is defined in `grammar.py`. The idea is quite similar to context-free grammars, but with an added *vary statement* which defines which preterminals and attributes to vary to generate the desired incorrect examples. See the CLAMS paper for more detail.

The generation procedure we use is defined in `generator.py`. We give the script a directory of grammars, wherein each file contains one syntactic test case. We also define a `common.avg` grammar for each language, which contains terminals shared by all other grammars in the directory. You can also check whether all tokens in your grammars are contained in your language model's vocabulary by using the `--check_vocab` argument, which takes a text file of line-separated tokens.

Example usage:
```
python generator.py --grammars fr_replication --check_vocab data/fr/vocab.txt
```

## Syntactic Evaluation Sets
The evaluation sets we use in the CLAMS paper are present in the `*_evalset` folders. They are formatted as tab-separated tables, where the first column is a Boolean representing the grammaticality of the sentence, and the second is the evaluation case.

Note that the AVGs generate examples which have a minimal amount of preprocessing---most tokens are lowercase, and by default, they do not contain punctuation or end-of-sentence markers. This is meant to keep them modular. We provide a `preproc.py` script which changes the format of the examples to better fit our training domain, and this should be modified to make the evaluation sets look more like your training sets (if you so choose). We use the `--eos` setting to obtain the results in Table 2 of our paper; we use both the `--eos` and `--capitalize` settings to obtain the results in Table 4. The `postproc.sh` script simply renames the files generated by `preproc.py` to replace the original un-processed files.

Example usage:
```
python preproc.py --evalsets fr_evalset --eos
./postproc.sh fr_evalset
```

## Language Model Training and Evaluation
### Training and Testing LMs
Requirements:
- Python 3.6.9+
- PyTorch 1.1.0
- CUDA 9.0

We modify the code of van Schijndel, Mueller & Linzen (2019), which itself is a modification of the code from Marvin & Linzen (2018). This code was written to run on a particular SLURM-enabled grid setup. We highly encourage pull requests containing code updated to run on more recent PyTorch/CUDA versions, as well as code meant to run on more types of systems.

To train an LSTM language model, run `train_{en,fr,de,ru,he}.sh` in `LM_syneval/example_scripts`.

To obtain model perplexities on a test corpus, run the following (in `LM_syneval/word-language-model`):
```
python main.py --test --lm_data $corpus_dir/ --save models/$model_pt --save_lm_data models/$model_bin --testfname $test_file
```
There is a test script in `LM_syneval/example_scripts`.

### Obtaining Word Scores on Syntactic Evaluation Sets

To obtain word-by-word model surprisals on the syntactic evaluation sets, run the following (in `LM_syneval/word-language-model`):
```
./test.sh $evalset_dir $model_dir $test_case
```
To evaluate on every test case in a directory of evaluation sets, pass `all` as the `$test_case` argument.

The above script outputs a series of files with the extension `.$model_dir.wordscores` in the same directory as the evaluation sets.

Then, to analyze these word-by-word scores and obtain scores per-case, run the following (in `LM_syneval/word-language-model`):
```
python analyze_results.py --score_dir $score_dir --case $case
```
where `$score_dir` is a directory containing `.wordscores` files, and `case` refers to the syntactic evaluation case (e.g., `obj_rel_across_anim`. By default, `--case` is `all`; this will give scores on every stimulus type in the specified directory.

By default, the above script compares the probability of entire grammatical and ungrammatical sentences when obtaining accuracies. To calculate accuracies based solely on the individual varied words, pass the `--word_compare` argument to `analyze_results.py`.


### Modified BERT-Syntax Code for mBERT
We provide a very slightly modified version of [Yoav Goldberg's BERT-Syntax code](https://github.com/yoavg/bert-syntax). Additionally, we provide scripts for pre-processing the syntactic evaluation sets generated by AVGs into the format required by BERT-Syntax.

The model loaded in `eval_bert.py` is now `bert-base-multilingual-cased`. Additionally, the script is now able to handle input other than cases from the English Marvin & Linzen set.

To pre-process an evaluation set for BERT or mBERT, copy the `make_for_bert.py` script to the folder containing the evaluation set and then run it from that directory. This will produce a `forbert.tsv` file which you can then pass as input to the `eval_bert.py` script.

Example usage:
```
python eval_bert.py marvin > results/$lang_results_multiling.txt
```

## Licensing
CLAMS is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.

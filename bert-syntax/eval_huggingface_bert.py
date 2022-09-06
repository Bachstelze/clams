from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Use multilingual model
model_name = 'bert-base-multilingual-cased'
multilingual_tokenizer = AutoTokenizer.from_pretrained(model_name)
multilingual_model = AutoModelForMaskedLM.from_pretrained(model_name, is_decoder=True)
#model.eval()

fill_pipeline = pipeline("fill-mask",
               model=multilingual_model,
               tokenizer=multilingual_tokenizer)

import sys
import csv
from collections import Counter

def get_probs_for_words(sent,w1,w2,result_file):
  pre,target,post=sent.split('***')
  scores = fill_pipeline(pre + multilingual_tokenizer.mask_token + post, targets=[w1, w2])

  try:
    return [scores[0]["score"], scores[1]["score"]]
  except IndexError:
    print("skipping",w1,w2,"bad wins",file=result_file)
    return None

def load_marvin(lang):
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = []
    for line in open(lang+"_forbert.tsv"):
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

def eval_marvin(language):
    result_file=open(language+'_result.txt', 'a')
    o = load_marvin(language)
    print(len(o),file=result_file)
    from collections import defaultdict
    import time
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i,(case,tp,s,g,b) in enumerate(o):
        ps = get_probs_for_words(s,g,b,result_file)
        if ps is None: ps = [0,1]
        gp = ps[0]
        bp = ps[1]
        print(gp>bp,case,tp,g,b,s,file=result_file)
        if i % 100==0:
            print(i,time.time()-start,file=result_file)
            start=time.time()
            sys.stdout.flush()
            
for lang in languages:
  eval_marvin(lang)

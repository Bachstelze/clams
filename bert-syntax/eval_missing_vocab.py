from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import sys
import csv
from collections import Counter

# Use multilingual model
model_name = 'bert-base-multilingual-cased'
multilingual_tokenizer = AutoTokenizer.from_pretrained(model_name)
multilingual_model = AutoModelForMaskedLM.from_pretrained(model_name, is_decoder=False)
#model.eval()

fill_pipeline = pipeline("fill-mask",
               model=multilingual_model,
               tokenizer=multilingual_tokenizer)

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

def get_probs_for_words(sent,w1,w2):
  score1 = get_single_word_score(sent,w1,w1,result_file,"",[])
  score2 = get_single_word_score(sent,w2,w2,result_file,"",[])

  if score1 is not None and score2 is not None:
    print(sent)
    print(w1 + " " +str(score1))
    print(w2 + " " +str(score2))
    return [score1, score2]

  print("skipping",w1,w2,"bad wins")
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
    print(len(o),file=sys.stderr)
    from collections import defaultdict
    import time
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i,(case,tp,s,g,b) in enumerate(o):
        ps = get_probs_for_words(s,g,b)
        if ps is None: ps = [0,1]
        gp = ps[0]
        bp = ps[1]
        print(gp>bp,case,tp,g,b,s)
        if i % 100==0:
            print(i,time.time()-start,file=sys.stderr)
            start=time.time()
            sys.stdout.flush()

def eval_lgd():
    for i,line in enumerate(open("lgd_dataset_with_is_are.tsv",encoding="utf8")):
        na,_,masked,good,bad = line.strip().split("\t")
        ps = get_probs_for_words(masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),na,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()


def read_gulordava():
    rows = csv.DictReader(open("generated.tab",encoding="utf8"),delimiter="\t")
    data=[]
    for row in rows:
        row2=next(rows)
        assert(row['sent']==row2['sent'])
        assert(row['class']=='correct')
        assert(row2['class']=='wrong')
        sent = row['sent'].lower().split()[:-1] # dump the <eos> token.
        good_form = row['form']
        bad_form  = row2['form']
        sent[int(row['len_prefix'])]="***mask***"
        sent = " ".join(sent)
        data.append((sent,row['n_attr'],good_form,bad_form))
    return data

def eval_gulordava():
    for i,(masked,natt,good,bad) in enumerate(read_gulordava()):
        if good in ["is","are"]:
            print("skipping is/are")
            continue
        ps = get_probs_for_words(masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),natt,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()
            
languages = ["de","en","ru","fr","he"]

if 'marvin' in sys.argv:
  for lang in languages:
    eval_marvin(lang)
elif 'gul' in sys.argv:
    eval_gulordava()
else:
    eval_lgd()

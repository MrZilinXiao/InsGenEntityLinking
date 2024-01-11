# usage: PYTHONPATH=src/ python EL_scripts/build_model_trie.py
import sys
import pickle
from tqdm import tqdm
# from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers import AutoTokenizer
from genre.trie import Trie
import re
import json
from unidecode import unidecode

def has_non_english_chars(string):
    return bool(re.search(r'[^\x00-\x7F]+', string))

tokenizer_name_or_path = "facebook/opt-125m"
dict_path = "/home/v-zilinxiao/code/transformers/src/genre/data/0501_mention_to_candidates_merge_gt.pkl"  # full mention list
out_file = "/home/v-zilinxiao/code/transformers/genre_additional_data/0511_{}_mention_trie_full.pkl".format(tokenizer_name_or_path.split('/')[-1])

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    # 0501 version has gt inside!
    # dict_path = "/home/v-zilinxiao/code/elevant/benchmarks/0420_eval_only_mention_lst_no_cap.json"  # do not remove unknown...
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    prefix_space = " " if 'llama' not in tokenizer_name_or_path.lower() else ""
    print("read mentions from %s..." % dict_path)
    if dict_path.endswith(".pkl"):
        with open(dict_path, "rb") as f:
            mention_candidates_dict = pickle.load(f)
    else:
        with open(dict_path, "r") as f:
            mention_candidates_dict = json.load(f)  # in fact no candidates... just mention and their appearance frequency
    print("build mention trie...")
    mention_trie = Trie()
    # for each mention?
    for i, mention in tqdm(enumerate(mention_candidates_dict), total=len(mention_candidates_dict)):
        try:
            # if dict_path.endswith(".pkl") and has_non_english_chars(mention):
            #     continue
            # encoded = tokenizer(" {}".format(mention)).input_ids[1:]  # skip BOS; but notice llama tokenizer does not need the head space
            encoded = tokenizer.encode("{}{}{}".format(prefix_space, mention, tokenizer.eos_token), add_special_tokens=False)
            if i < 10 or len(mention_candidates_dict) - 10 < i < len(mention_candidates_dict) :
                print(encoded, mention)
            # encoded = model.encode(" {}".format(mention))[1:].tolist()
        except (UnicodeEncodeError, TypeError) as e:
            print(f"{str(e)}: {i}")
            continue
        mention_trie.add(encoded)
        if dict_path.endswith(".json") and has_non_english_chars(mention):
            print(f"non-English mention: {mention}, ", end='')
            ascii_mention = unidecode(mention)
            print(f"adding ascii version: {ascii_mention}...")
            encoded = tokenizer.encode("{}{}{}".format(prefix_space, ascii_mention, tokenizer.eos_token), add_special_tokens=False)
            mention_trie.add(encoded)
    # out_file = "/home/v-zilinxiao/code/transformers/genre_additional_data/llama_mention_trie%s.pkl" % (".dalab-entities" if dalab_entities else "")
    # out_file = "/home/v-zilinxiao/code/transformers/genre_additional_data/llama_mention_trie_fixed_space_filter_non_English.pkl"
    
    print("save trie to %s..." % out_file)
    with open(out_file, "wb") as f:
        pickle.dump(mention_trie, f)

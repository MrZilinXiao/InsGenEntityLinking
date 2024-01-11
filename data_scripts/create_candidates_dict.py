import re
import string
import itertools
import pickle
import sys
import json
from unidecode import unidecode

eval_mention_lst = json.load(open('/home/v-zilinxiao/code/elevant/benchmarks/0420_eval_only_mention_lst_no_cap_remove_unknown.json', 'r'))

def read_dalab_candidates():
    for line in open("data/dalab/prob_yago_crosswikis_wikipedia_p_e_m.txt"):
        line = line[:-1]
        columns = line.split("\t")
        mention = columns[0]
        for column in columns[2:]:
            if len(column.strip()) == 0:
                continue
            values = column.split(",")
            candidate = ",".join(values[2:])
            candidate = candidate.replace("_", " ")
            yield mention, candidate


def hex2int(hexa: str) -> int:
    return int(hexa, 16)

def has_non_english_chars(string):
    return bool(re.search(r'[^\x00-\x7F]+', string))

def replace_unicode(u_str):
    matches = set(re.findall("\\\\u....", u_str))
    for match in matches:
        u_str = u_str.replace(match, chr(hex2int(match[2:])))
    return u_str


PUNCTUATION_CHARS = set(string.punctuation)


def filter_mention(mention):
    # if mention[0].islower():
    #     return True
    # if mention.isdigit():
    #     return True
    if mention[0] in PUNCTUATION_CHARS:
        return True
    return False


def read_aida_candidates():
    for line in open("data/aida/aida_means.tsv"):
        line = line[:-1]
        values = line.split("\t")
        mention = replace_unicode(values[0][1:-1])
        candidate = replace_unicode(values[1]).replace("_", " ")
        yield mention, candidate


def read_entities_universe():
    entities = set()
    for line in open("data/dalab/entities_universe.txt"):
        entity = line.split("\t")[-1][:-1].replace("_", " ")
        entities.add(entity)
    return entities


if __name__ == "__main__":
    dalab_entities = "--dalab" in sys.argv
    if dalab_entities:
        print("read entities universe...")
        entities = read_entities_universe()
    mention_candidates_dict = {}
    print("read mention - candidate pairs...")
    n = 0
    for mention, candidate in itertools.chain(read_dalab_candidates(), read_aida_candidates()):  
        # combine both dalab and aida mention-candidate
        if filter_mention(mention):  # skipped lower and punctuation
            # update on 04/08: keep lower
            continue
        if dalab_entities and mention not in entities:
            continue
        if mention not in mention_candidates_dict:
            mention_candidates_dict[mention] = set()
        mention_candidates_dict[mention].add(candidate)
        n += 1
        if n % 10000 == 0:
            print("\r%i pairs" % n, end="")
    print()
    n = 0
    
    # scan over mention_candidates_dict and merge possible latin entries
    mention_keys = list(mention_candidates_dict.keys())
    for mention in mention_keys:
        if has_non_english_chars(mention):
            ascii_mention = unidecode(mention)
            if ascii_mention in mention_candidates_dict:
                tmp = list(mention_candidates_dict[ascii_mention])
                mention_candidates_dict[ascii_mention].update(mention_candidates_dict[mention])
                mention_candidates_dict[mention].update(tmp)
            else:
                mention_candidates_dict[ascii_mention] = mention_candidates_dict[mention]  # direct copy
        n += 1
        if n % 10000 == 0:
            print("\r%i pairs" % n, end="")
            
    for mention in mention_candidates_dict:
        mention_candidates_dict[mention] = sorted(mention_candidates_dict[mention])
    out_file = "data/0427_mention_to_candidates_merge_latin_dict%s.pkl" % (".dalab-entities" if dalab_entities else "")
    print("write to %s..." % out_file)
    with open(out_file, "wb") as f:
        pickle.dump(mention_candidates_dict, f)
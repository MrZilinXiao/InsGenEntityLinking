import sys
import pickle
from tqdm import tqdm
from genre.fairseq_model import GENRE
from genre.trie import Trie


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    dalab_entities = "--dalab" in sys.argv
    model_path = "models/fairseq_e2e_entity_linking_wiki_abs"
    model = GENRE.from_pretrained(model_path).eval()
    dict_path = "data/mention_to_candidates_dict%s.pkl" % (".dalab-entities" if dalab_entities else "")
    print("read mentions from %s..." % dict_path)
    with open(dict_path, "rb") as f:
        mention_candidates_dict = pickle.load(f)
    print("build mention trie...")
    mention_trie = Trie()
    for mention in tqdm(mention_candidates_dict):  # google: candidates
        try:
            encoded = model.encode(" {}".format(mention))[1:].tolist()
        except UnicodeEncodeError:
            continue
        mention_trie.add(encoded)
    out_file = "data/mention_trie%s.pkl" % (".dalab-entities" if dalab_entities else "")
    print("save trie to %s..." % out_file)
    with open(out_file, "wb") as f:
        pickle.dump(mention_trie, f)

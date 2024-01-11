import argparse
import json
import urllib.parse
import pickle


def create_label_json(begin, end, qid):
    return {
        "span": [begin, end],
        "recognized_by": "GENRE",
        "id": qid,
        "linked_by": "GENRE",
        "candidates": [qid]
    }


def compute_labels(paragraph: str, labeled_paragraph: str, start_position: int):
    if paragraph == labeled_paragraph:
        return []
    p_pos = 0
    l_pos = 0
    start = end = 0
    labels = []
    while p_pos < len(paragraph):
        p_char = paragraph[p_pos]
        if p_char in " []\n":
            p_pos += 1
            continue
        l_char = labeled_paragraph[l_pos]
        if l_char in " \n":
            l_pos += 1
        elif l_char == p_char:
            l_pos += 1
            p_pos += 1
            end = p_pos
        elif l_char == "{":
            start = p_pos
            l_pos += 1
        elif l_char == "}":
            label_start = l_pos + 3
            label_end = labeled_paragraph.find("]", label_start)
            label = labeled_paragraph[label_start:label_end].strip()
            l_pos = label_end + 1
            labels.append((start_position + start, start_position + end, label))
        else:
            print("WARNING:", p_pos, l_pos, p_char, l_char)
            break
    return labels


def get_mapping():
    prefix = "https://en.wikipedia.org/wiki/"
    mapping = {}
    for line in open("data/elevant/qid_to_wikipedia_url.tsv"):
        line = line[:-1]
        vals = line.split("\t")
        qid = vals[0]
        wikipedia_title = urllib.parse.unquote(vals[1][len(prefix):]).replace("_", " ")
        mapping[wikipedia_title] = qid
    return mapping


def main(args):
    if not args.wikipedia:
        print("read mapping...")
        mapping = get_mapping()

        print("load redirects...")
        with open("data/elevant/link_redirects.pkl", "rb") as f:
            redirects = pickle.load(f)

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_file[:-4] + ".jsonl"

    with open(output_file, "w") as out_file:
        for line in open(args.input_file):
            data = json.loads(line)
            print("== " + str(data["id"]) + " (" + str(data["evaluation_span"]) + ") ==")
            text = data["text"]
            genre_text = data["GENRE"]
            position = 0

            labels = []
            wikipedia_labels = compute_labels(text, genre_text, position)
            for start, end, label in wikipedia_labels:
                qid = label
                if args.wikipedia:
                    qid = "https://en.wikipedia.org/wiki/" + label.replace(" ", "_")
                else:
                    if label in mapping:
                        qid = mapping[label]
                    elif label in redirects:
                        redirected = redirects[label]
                        if redirected in mapping:
                            qid = mapping[redirected]
                print(start, end, label, qid)
                labels.append(create_label_json(start, end, qid))
            data["entity_mentions"] = labels
            out_file.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_file", type=str)
    parser.add_argument("-o", dest="output_file", type=str, default=None)
    parser.add_argument("--wikipedia", action="store_true")
    args = parser.parse_args()
    main(args)

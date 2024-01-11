# from Wikidump to jsonl file: add mention and entity boundaries
import random
from wikipedia2vec.dump_db import DumpDB
import json
from tqdm import tqdm
import re
from contextlib import closing
from multiprocessing.pool import Pool
import sys
import os
os.environ['PYTHONPATH'] = "/home/v-zilinxiao/code/transformers/src"
sys.path.insert(0, "/home/v-zilinxiao/code/transformers/src")
import transformers
from common_utils.sentence_splitter import OpenNLPSentenceSplitter
from transformers.models.llama.tokenization_llama import LlamaTokenizer
print(f"transformers version: {transformers.__version__}")  # should end with dev0
assert transformers.__version__.endswith("dev0")

# global vars usd in pool workers
_dump_db = _tokenizer = _sentence_splitter = None


def preprocess_text(text: str) -> str:
    # preprocess replace must not change index!
    return text.replace("\xa0", " ").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")").replace("\n", " ")


class WikiDumper:
    def __init__(self):
        # self.dump_db = DumpDB(dump_db_path)  # shared across processes
        pass

    @staticmethod
    def _init_worker(
        dump_db,
        tokenizer,
        sentence_splitter
    ):
        global _dump_db, _tokenizer, _sentence_splitter
        _dump_db = dump_db
        _tokenizer = tokenizer
        _sentence_splitter = sentence_splitter

    # update on 04/27: turn into paragraph-level processing rather than page-level
    @staticmethod
    def process_page(page_title, debug=False):
        # read in the page, output the input_seq and output_seq with mention identifiers

        # mention_entity: [{'mention_str':..., 'entity':..., 'src_span':...}, ...]
        ret = []
        seen_mention_dict = {}
        # skip redirect and disambiguation pages
        if _dump_db.is_redirect(page_title) or _dump_db.is_disambiguation(page_title):
            return ret
        
        for kk, paragraph in enumerate(_dump_db.get_paragraphs(page_title)):
            if not paragraph.abstract:
                continue
            
            sentences = []
            ori_sentences = []
            mention_entity = []  # save entry-level mention_entity mapping
            
            paragraph_text = preprocess_text(paragraph.text)  # preprocess here
            
            if len(paragraph_text.split()) < 10:   # filter short paragraphs
                continue
            
            # First, get paragraph links.
            # Paragraph links are represented its form (link_title) and the start/end positions of strings
            # (link_start, link_end).
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)  # resolve redirect
                mention_str = paragraph_text[link.start: link.end]
                assert mention_str == preprocess_text(link.text), f"mention_str: {mention_str}, link.text: {link.text}"
                # remove category links
                if (
                    link_title.startswith("Category:")
                    # and link.text.lower().startswith("category:")
                    or link_title.startswith("File:")
                ):  # rebuild paragraph text to skip category mention in the page
                    paragraph_text = (
                        paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                    )  # remove links to category pages; set to empty is a good choice?
                else:
                    # a good mention; if you want to filter specific mentions, do it here.
                    if link.text:
                        paragraph_links.append((link.text, link_title, link.start, link.end))
            
            # filter strange paragraphs with no links or no context text!
            if not paragraph_links:
                continue
            
            mention_len = sum(link[3] - link[2] for link in paragraph_links)
            if mention_len > len(paragraph_text) * 0.8:
                continue

            # add string heruistics match to add co-reference
            for link_text, link_title, _, _ in paragraph_links:
                # if link_text:
                if link_text not in seen_mention_dict:
                    seen_mention_dict[link_text] = link_title  # added if appears for the first time
                elif seen_mention_dict[link_text] is None:
                    continue  # mark, never added again
                elif seen_mention_dict[link_text] != link_title:  # this is unusal, but it happens when the same mention string is used for different entities
                    if debug:
                        print(f"WARNING: duplicate and inconsistent mention `{link_text}` found in the page: ", paragraph_text)
                    seen_mention_dict[link_text] = None  # mark, never added again

            # search over this paragraph, skip the link itself and any nested paragraph links
            for seen_mention, tgt_title in seen_mention_dict.items():
                if tgt_title is None:
                    continue
                for match in re.finditer(r"\b" + re.escape(seen_mention) + r"\b", paragraph_text):  # !!!full word match!!!
                    # match.start() and match.end() can not be in any of the paragraph_links
                    skip = False
                    for _, _, link_start, link_end in paragraph_links:
                        if (
                            (match.start() >= link_start and match.end() <= link_end)  # nested inside
                            or (link_end >= match.start() >= link_start and match.end() >= link_end)  # partially overlap
                            or (link_start >= match.start() and match.end() >= link_start)  # partially overlap
                        ):
                            skip = True
                            break
                        # good co-reference mentions are kept
                    if not skip:
                        if debug:
                            print(f"found co-reference mention: {(seen_mention, tgt_title, match.start(), match.end())}, paragraph_links: {paragraph_links}")
                        paragraph_links.append((seen_mention, tgt_title, match.start(), match.end()))

            # must sort the paragraph_links by start position
            paragraph_links = list(sorted(paragraph_links, key=lambda x: x[2]))

            sent_spans = _sentence_splitter.get_sentence_spans(paragraph_text.rstrip())

            for sent_start, sent_end in sent_spans:  # for each sentence in this paragraph
                sent_words = ""
                ori_sent_words = ""
                cur = sent_start
                sent_links = [(link_title, link_start, link_end) for _, link_title, link_start, link_end in paragraph_links
                              if sent_start <= link_start < sent_end and link_end <= sent_end]

                # print(sent_links)

                if len(sent_links) == 0:
                    sent_words = paragraph_text[sent_start:sent_end]
                    sentences.append(sent_words)
                    ori_sentences.append(sent_words)
                    continue

                for link_title, link_start, link_end in sent_links:
                    sent_words += paragraph_text[cur:link_start]  # add words before this link in tgt_seq
                    # add a leading space if there is not one
                    if link_start > sent_start and paragraph_text[link_start - 1] != " ":
                        sent_words += " "
                    sent_words += "[ " + paragraph_text[link_start:link_end] + " ]"
                    sent_words += " { " + link_title + " }"  # add entity
                    # add a tailing space if there is not one
                    if link_end < sent_end and paragraph_text[link_end] != " ":
                        sent_words += " "

                    # link_start, link_end is based on this paragraph, so when adding to mention_entity, we need to add an offset of previous sentences
                    mention_entity.append({
                        'mention_str': paragraph_text[link_start:link_end],
                        'entity': link_title,
                        # 'src_span': (link_start + real_offset, link_end + real_offset),  # +1 or not? check later
                        # 'link_span': (link_start, link_end)
                    })

                    ori_sent_words += paragraph_text[cur:link_end]  # add words before the mention
                    cur = link_end
                # attach last part
                sent_words += paragraph_text[cur:sent_end]
                ori_sent_words += paragraph_text[cur:sent_end]

                sentences.append(sent_words)
                ori_sentences.append(ori_sent_words)
                
            src_seq = " ".join(ori_sentences)   # paragraph_text is not identical with src_seq, leading to some index error
            tgt_seq = " ".join(sentences)
            if not src_seq or not tgt_seq:
                continue
            # as the mention_entity are organized in order, we ensure the first hit by string.find is the correct mention
            src_st = 0
            for i in range(len(mention_entity)):
                offset = src_seq[src_st:].find(mention_entity[i]['mention_str'])
                assert offset != -1, f"mention string `{mention_entity[i]['mention_str']}` not found in src_seq: {src_seq}"
                mention_entity[i]['src_span'] = (src_st + offset, src_st + offset + len(mention_entity[i]['mention_str']))
                src_st = src_st + offset + len(mention_entity[i]['mention_str'])
            ret.append({
                'page_title': page_title,
                'para_id': kk, 
                'src': src_seq,
                'tgt': tgt_seq,
                'mention_entity': mention_entity,
                'llama_len': len(_tokenizer.tokenize(src_seq + " " + tgt_seq))
            })
        return ret


    @classmethod
    def build(
        cls,
        dump_db,
        tokenizer,
        sentence_splitter,
    ):
        target_titles = [
            title
            for title in dump_db.titles()
            if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))  # no ':' or no image/file/category -> target titles available
        ]

        random.shuffle(target_titles)
        with open("0220-full-para-filtered.jsonl", 'w') as f:
            with tqdm(total=len(target_titles)) as pbar:
                init_args = (
                    dump_db,
                    tokenizer,
                    sentence_splitter
                )
                with closing(
                    Pool(40, initializer=WikiDumper._init_worker, initargs=init_args)
                ) as pool:
                    # for source, target in pool.imap(WikiDumper.process_page, target_titles, chunksize=100):
                    for ret in pool.imap(WikiDumper.process_page, target_titles, chunksize=100):
                        # source_f.write(source + "\n")
                        # target_f.write(target + "\n")
                        if ret is not None:
                            for sample in ret:
                                f.write(json.dumps(sample) + "\n")
                        pbar.update()
        print(f"Done! {len(target_titles)} articles processed.")


if __name__ == '__main__':
    print(f"Loading necessary resources...")
    dump_db = DumpDB("/home/v-zilinxiao/data/dataset/wiki_raw/0220-full.db")
    tokenizer = LlamaTokenizer.from_pretrained("/home/v-zilinxiao/code/transformers/llama_7B_0.5_lora_spacing/checkpoint-1000")
    sentence_splitter = OpenNLPSentenceSplitter()

    WikiDumper.build(dump_db, tokenizer, sentence_splitter)

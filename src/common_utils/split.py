import re
from common_utils.sentence_splitter import OpenNLPSentenceSplitter

opennlp = None

def opennlp_split_sentence(text, keep_index=True):
    global opennlp
    if opennlp is None:
        opennlp = OpenNLPSentenceSplitter()
    
    st_ed_lst = opennlp.get_sentence_spans(text)
    if not keep_index:
        return [text[st: ed] for st, ed in st_ed_lst]
    st_lst = [st for st, _ in st_ed_lst] + [len(text)]
    # return splitted sentences only by st; when keep_index, do not join by space.
    return [text[st_lst[i-1]: st_lst[i]] for i in range(1, len(st_lst))]

def split_sentence(spacy_model, text):
    # split text into sentences
    doc = spacy_model(text)
    sentences = [sent.text for sent in doc.sents]  # spacy model truncates text into custom parts
    return sentences

# _split_long_texts by sentence, most elegant way;
# also try not splitting as llama model uses rotatE;

def split_by_punc_fn(text):
    pattern = re.compile(r'(?<=[^\s\w])\s+')
    return pattern.split(text)


def split_text_to_parts(spacy_model, text, tokenizer,
                        max_allowed_tokens_each_part=175, split_type='average', 
                        split_by_punc=False, split_by_space=False):
    # split_type average is `iterative`, making sure each part has similar length
    if split_by_space:
        sentences = text.split()
    elif split_by_punc:
        sentences = split_by_punc_fn(text)
    else:
        sentences = split_sentence(spacy_model, text)
    
    if split_type == 'sent':  # submit by sent; fall back to punc only when too long
        for sent in sentences:  # just check! 
            selected_token_num = len(tokenizer(sent).input_ids[1:])
            if not split_by_punc and selected_token_num > 2 * max_allowed_tokens_each_part:
                print(f"\nWARNING: selected_token_num {selected_token_num} too extreme, fall back to split_by_punc")
                return split_text_to_parts(spacy_model, text, tokenizer,
                                            max_allowed_tokens_each_part, split_type=split_type, 
                                            split_by_punc=True, split_by_space=False)
            if not split_by_space and selected_token_num > 2 * max_allowed_tokens_each_part:
                # too extreme, we just fall back to split_by_space
                print(f"\nWARNING: selected_token_num {selected_token_num} too extreme, fall back to split_by_space")
                return split_text_to_parts(spacy_model, text, tokenizer,
                                            max_allowed_tokens_each_part, split_type=split_type, 
                                            split_by_punc=False, split_by_space=True)
                
        return sentences, len(sentences)
            
    
    if split_type == 'average':
        n_parts = 1
        while n_parts <= len(sentences):
            print(f"\rtrying {n_parts} parts for {len(sentences)} sentences", end="", flush=True)
            sents_per_part = len(sentences) / n_parts  # in float, later round
            split_results = []
            did_failed = False
            for i in range(n_parts):
                sent_st = int(i * sents_per_part)
                sent_ed = int((i + 1) * sents_per_part)
                selected_part = " ".join(sentences[sent_st: sent_ed])
                selected_token_num = len(tokenizer(selected_part).input_ids[1:])
                if selected_token_num > max_allowed_tokens_each_part:
                    result = None
                else:
                    result = selected_part
                if result is not None:
                    split_results.append(result)
                elif sent_ed - sent_st == 1:
                    if not split_by_punc and selected_token_num > 2 * max_allowed_tokens_each_part:
                        # too extreme, we just fall back to split_by_punc
                        print(f"\nWARNING: selected_token_num {selected_token_num} too extreme, fall back to split_by_punc")
                        return split_text_to_parts(spacy_model, text, tokenizer,
                                                   max_allowed_tokens_each_part, split_type=split_type, 
                                                   split_by_punc=True, split_by_space=False)
                    if not split_by_space and selected_token_num > 2 * max_allowed_tokens_each_part:
                        # too extreme, we just fall back to split_by_space
                        print(f"\nWARNING: selected_token_num {selected_token_num} too extreme, fall back to split_by_space")
                        return split_text_to_parts(spacy_model, text, tokenizer,
                                                   max_allowed_tokens_each_part, split_type=split_type, 
                                                   split_by_punc=False, split_by_space=True)
                    else:
                        split_results.append(selected_part)  # despite exceeding max_tokens, this is the minimum part!
                else:
                    did_failed = True
                    break
            if did_failed:
                n_parts += 1
            else:
                return split_results, len(sentences)
    # now for clip type
    elif split_type == 'clip':
        raise NotImplementedError(f"clip type is deprecated, use average instead")
        # split_results = []
        # part = ""
        # num_tokens = 0
        # for sent in sentences:
        #     sent_num_tokens = len(tokenizer(sent).input_ids[1:])
        #     if len(part) > 0 and num_tokens + sent_num_tokens > max_allowed_tokens_each_part:
        #         split_results.append(part)
        #         part = ""
        #         num_tokens = 0
        #     if len(part) > 0:
        #         part += " "

        #     part += sent
        #     num_tokens += sent_num_tokens
        # if len(part) > 0:
        #     split_results.append(part)
        # return split_results, len(sentences)
    else:
        raise ValueError(f"split_type {split_type} not supported")
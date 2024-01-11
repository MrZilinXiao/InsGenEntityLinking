# abandoned, not referred anywhere
from typing import Dict, List
import torch
from genre.trie import DummyTrieEntity, DummyTrieMention, Trie


STATUS_TO_STRING = {
    'o': 'Outside of Mention/Entity', 
    'm': 'Inside of Mention', 
    'e': 'Inside of Entity',
}

def get_prefix_allowed_tokens_fn_universal(  # we hope it could quickly adapt to any other models;
    tokenizer,
    sentences: List[str],
    prompt_sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: tokenizer.encode(x, add_special_tokens=False),
        lambda x: tokenizer.decode(torch.tensor(x)),
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.eos_token,  # for manually adding eos
        len(tokenizer) - 1,
        sentences,
        prompt_sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
        debug, 
        tokenizer.name_or_path,
    )
    
    
def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    eos_token,   # for manually adding eos
    vocabulary_length,
    sentences: List[str],  # sentences to be linked
    prompt_sentences: List[str],  # sentences to be used as instructions
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False, 
    model_name="llama_placeholder",
):
    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"
    prefix_space = " " if 'llama' not in model_name.lower() else ""  # for llama, we do not need space;
    codes = {
        # n: encode_fn(" {}".format(c))[1]
        n: encode_fn("{}{}".format(prefix_space, c))[0]  # when encoding, do not add offset;
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id
    
    if mention_trie is None:
        # no constraint on mentions
        mention_trie = DummyTrieMention(  # dummy trie returns a sequence of all tokens except bos + pad -> no constraint on mentions
            # **however pad_token_id is identical to eos_token_id in llama**
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )
        
    if candidates_trie is None and mention_to_candidates_dict is None:
        # no constraint on candidates; we do not use this
        candidates_trie = DummyTrieEntity(  # dummy trie returns a sequence of all tokens except bos + pad -> no constraint on candidates
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ], 
            codes,
        )
    
    # remove [1:] as now all encode_fn will not prepend bos
    sents_ori = [[codes["EOS"]] + encode_fn(sent) for sent in sentences]
    sent_prompt_lens = [len(encode_fn(sent)) + 1 for sent in prompt_sentences]  # +1 compensate for bos
    
    if debug:
        print(f"sent_origs prepared: {sents_ori}")
        print(f"sent_prompt_lens prepared: {sent_prompt_lens}")
        
    # help function deciding current generation position below
    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        status = sum(e in c for e in sent) % 4
        # 0 -> outside any mention or entity (passed through 4x boundary tokens);
        # 1 -> mention;
        # 2 -> entity about to begin;
        # 3 -> in entity;

        if status == 0:
            return "o"
        elif status == 1:
            return "m"
        else:
            return "e"
        
    def get_pointer_end(sent, sent_orig):
        # return the index of the last token_id in the generated sentence that matches sent_orig
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):  # skip mention boundary tokens
                i += 1
            elif sent[i] == codes["start_entity_token"]:  # skip entity identifier in generated sentence
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None
    
        
    def get_trie_outside(sent, sent_orig):
        pointer_end = get_pointer_end(sent, sent_orig)  # get current generated position idx
        # if debug:
        #     print(f'trie_outside sent: {sent}')
        #     print(f'trie_outside sent_orig: {sent_orig}')
        #     print(f"pointer_end: {pointer_end}")

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"] and sent_orig[
                pointer_end
            ] in mention_trie.get([]):
                return [sent_orig[pointer_end], codes["start_mention_token"]]
                # when 1. ref sent not ending, 2. current ref token in mention allow list -> allow generating a mention start
            else:
                return [sent_orig[pointer_end]]  # otherwise, only allow identical generation
        else:
            return []
        

    def get_trie_mention(sent, sent_orig):
        if debug:
            print(f'trie_mention sent: {sent}')
            print(f'trie_mention sent_orig: {sent_orig}')
        pointer_start, _ = get_pointer_mention(sent)  # find mention start and end in generated sentence
        if pointer_start + 1 < len(sent):  # if mention start is not just generated
            ment_next = mention_trie.get(sent[pointer_start + 1 :])  
            # put already generated mention seq in, get next possible tokens using trie
        else:
            ment_next = mention_trie.get([])  # the mention is just generated, possible next tokens could be anything
            # however, do not allow immediate close this mention, so eos is not in this branch

        pointer_end = get_pointer_end(sent, sent_orig)  
        # return the index of the last token_id in the generated sentence that matches sent_orig

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"]:
                if sent_orig[pointer_end] in ment_next:
                    if codes["EOS"] in ment_next: # if generated mention is valid, can continue generate mention span or close mention
                        return [sent_orig[pointer_end], codes["end_mention_token"]]  
                    else:  # incomplete mention, can only continue generate next token
                        return [sent_orig[pointer_end]]
                # WARNING: in the middle of some words, we can stop a mention, like [Apple] {Apple Inc.} 's iPhone
                
                elif codes["EOS"] in ment_next:  
                    # can not continue generate mention span (as sent_orig[pointer_end] not in ment_next)
                    # -> only to close the mention
                    return [codes["end_mention_token"]]
                
                else:
                    return []  # current generated mention is invalid;  
                # TODO: can this allow automatic fallback to next search beam? otherwise we should implement manual fallback
            else:  # when sent_orig pointer already hits EOS, we can only generate end_mention_token
                return [codes["end_mention_token"]]
        else:
            return []

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_entity(sent, sent_orig):
        pointer_start, pointer_end = get_pointer_mention(sent)  # do not care sent_orig

        if pointer_start + 1 != pointer_end:  # valid mention span
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()  # mention_str
            if debug:
                print(f"mention string: {mention}")

            if candidates_trie is not None:  # usually we don't give this
                candidates_trie_tmp = candidates_trie
            elif mention_to_candidates_dict is not None:  # dynamic construct trie of all candidates when needed
                if debug:
                    print(f"candidates of this mention: {mention_to_candidates_dict.get(mention, ['NIL'])}")
                candidates_trie_tmp = Trie(
                    [
                        # encode_fn(
                        #     " {} {} {} {}".format(
                        #         end_mention_token,
                        #         start_entity_token,
                        #         e,
                        #         end_entity_token,
                        #     )
                        # )[1:]
                        encode_fn(
                            # "{} {} {} {}</s>".format(  # add EOS manually
                            "{}{} {} {} {}{}".format(
                                prefix_space,  # except llama, other models need prefix space to encode the correct boundary
                                end_mention_token,
                                start_entity_token,
                                e,
                                end_entity_token,
                                eos_token,
                            )
                        )
                        for e in mention_to_candidates_dict.get(mention, ["NIL"])
                    ]
                )
            else:
                raise RuntimeError()

            return candidates_trie_tmp.get(sent[pointer_end:])

        return []
        
    def prefix_allowed_tokens_fn(batch_id, sent):  # receive a batch of sentences from hf
        sent = sent.tolist()  # **tokenized generated sentence**; consists of prompt parts!
        sent = [eos_token_id] + sent[sent_prompt_lens[batch_id]:]   # remove prompt parts while keep eos
        
        if debug:
            print(f"cropped generated sentence (with eos start): {sent}\ncropped generated sentence: {decode_fn(sent)}")
            
        status = get_status(sent)
        
        if debug:
            print(f"status: {STATUS_TO_STRING[status]}")
            
        sent_orig = sents_ori[batch_id]
        
        if status == "o":
            # passed, will not return int
            trie_out = get_trie_outside(sent, sent_orig)
        elif status == "m":
            # passed, will not return int
            trie_out = get_trie_mention(sent, sent_orig)  # if trie_out -> [], will predict on `0`
        elif status == "e":
            # possibly return an int
            trie_out = get_trie_entity(sent, sent_orig)
            if trie_out == codes["EOS"]:  # that special </s> will trigger outside generation
                print(f"CAUTION: trie_out is EOS, so we are outside!")
                trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError
        
        if isinstance(trie_out, int):  # cover corner case
            trie_out = [trie_out]
            print(f"WARNING: trie_out is int: {trie_out}")
            
        if debug:
            print(f"trie_out: {trie_out}, ", end='')
            print(f"which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")
            print('------------------------')

        return trie_out  # all allowed generated position

    return prefix_allowed_tokens_fn


if __name__ == "__main__":
    # TODO: unittests later covering all used tokenizers
    pass
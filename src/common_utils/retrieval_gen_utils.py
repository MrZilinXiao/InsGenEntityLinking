from ctypes import pointer
from typing import List, Dict
from numpy import add

from transformers import StoppingCriteria
import torch
from genre.entity_linking import Trie

def get_retrieval_prefix_allowed_tokens_fn_mention(
    tokenizer,
    mention_lst: List[str],
    truc_sentence: str,  # tructated until the span end, useful for guiding original sentence output
    prompt_len: int,
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    possible_mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False
):
    return _get_retrieval_prefix_allowed_tokens_fn_mention(
        lambda x: tokenizer.encode(x, add_special_tokens=False),
        lambda x: tokenizer.decode(torch.tensor(x), skip_special_tokens=False),
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.eos_token,  # for manually adding eos
        truc_sentence,
        mention_lst,
        prompt_len,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        possible_mention_to_candidates_dict,
        debug, 
        tokenizer.name_or_path,
    )


def _get_retrieval_prefix_allowed_tokens_fn_mention(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    eos_token,   # for manually adding eos
    # vocabulary_length,
    truc_sentence: str,  # note truc_sentence may contain unexcepted spaces...
    mention_lst: List[str],
    prompt_len: int,
    start_mention_token: str,
    end_mention_token: str,
    start_entity_token: str,
    end_entity_token: str,
    possible_mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False, 
    model_name="llama_placeholder",
):
    # available options are:
    prefix_space = " " if 'llama' not in model_name.lower() else ""  # for llama, we do not need space;
    this_mention_trie = Trie([
        encode_fn("{}{}{}".format(prefix_space, mention, eos_token))
        for mention in mention_lst
    ])  # can allow beam search over mention disambiguation
    # note that mention_trie can be triggered in any range within the selected span
    
    codes = {
        n: encode_fn("{}{}".format(prefix_space, c))[0]
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
    
    sent_ori = [codes["EOS"]] + encode_fn(truc_sentence)
    
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
        
    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end
        
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
            elif sent[i] == codes["start_entity_token"]:  
                # skip entity identifier in generated sentence
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None
    
            
    def get_trie_outside(sent, sent_ori):
        pointer_end = get_pointer_end(sent, sent_ori)
        if debug:
            print(f'trie_outside sent: {sent}')
            print(f'trie_outside sent_ori: {sent_ori}')
            print(f"trie_out pointer_end: {pointer_end}")
        if pointer_end:
            if sent_ori[pointer_end] != codes["EOS"] and \
                sent_ori[pointer_end] in this_mention_trie.get([]):
            # a) reference sentence not ending; b) this token can be a start of possible mentions
                return [sent_ori[pointer_end], codes['start_mention_token']]
            
            else:  # only allow identical generation
                return [sent_ori[pointer_end]]
            
        else:
            return []
        
    def get_trie_mention(sent, sent_orig):
        if debug:
            print(f'trie_mention sent: {sent}')
            print(f'trie_mention sent_orig: {sent_orig}')
        pointer_start, _ = get_pointer_mention(sent)  # find mention start and end in generated sentence
        if pointer_start + 1 < len(sent):  # if mention start is not just generated
            ment_next = this_mention_trie.get(sent[pointer_start + 1 :])  
            # put already generated mention seq in, get next possible tokens using trie
        else:
            ment_next = this_mention_trie.get([])  # the mention is just generated, possible next tokens could be anything
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
        
    def get_trie_entity(sent, _):  # when generating entity, we do not care about the original sentence
        pointer_start, pointer_end = get_pointer_mention(sent)
        if pointer_start + 1 != pointer_end:
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()
            if debug:
                print(f"mention plain-text: {mention}")
            
            assert possible_mention_to_candidates_dict is not None, f"possible_mention_to_candidates_dict is None"
            assert mention in possible_mention_to_candidates_dict, f"mention `{mention}` not in possible_mention_to_candidates_dict"
            if debug:
                print(f"possible candidates: {possible_mention_to_candidates_dict[mention]}")
            
            candidates_trie_tmp = Trie([
                encode_fn(
                    "{}{} {} {} {}{}".format(
                        prefix_space,  # except llama, other models need prefix space to encode the correct boundary
                        end_mention_token,
                        start_entity_token,
                        e,
                        end_entity_token,
                        eos_token,
                    )
                )
                for e in possible_mention_to_candidates_dict[mention]
            ])
            
            return candidates_trie_tmp.get(sent[pointer_end:])
        
        return []
            
            
    def prefix_allowed_tokens_fn(batch_id, sent):
        # assert sent.shape[0] == 1, f"now only support batch_size=1, get sent.shape == {sent.shape}"
        assert batch_id == 0, f"now only support batch_size=1, get batch_id == {batch_id}"
        sent = sent.tolist()  # sent is a list of generated token ids
        # remove irrelavent prompt tokens
        sent = [eos_token_id] + sent[prompt_len:]

        if debug:
            print(f"cropped generated sentence (with eos start): {sent}")
            print(f"decoded: `{decode_fn(sent)}`")
        
        status = get_status(sent)
        
        if debug:
            print(f"status: {status}")
        
        if status == "o":  # outside any mention or entity
            trie_out = get_trie_outside(sent, sent_ori)
            
        elif status == "m":
            trie_out = get_trie_mention(sent, sent_ori)
            
        elif status == "e":
            trie_out = get_trie_entity(sent, sent_ori)
            # codes["EOS"] should not be triggered
        
        else:
            raise RuntimeError
        
        if debug:
            print(f"trie_out: {trie_out}, ", end='')
            print(f"which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")
            print('------------------------')
            
        return trie_out
    
    return prefix_allowed_tokens_fn


class SelectedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, judge_by_token_id=True, 
                 start_mention_token="{", end_mention_token="}",
                 start_entity_token="[", end_entity_token="]", 
                 start_length=None, this_mention_str=None, debug=False, ins_length: int = None) -> None:
        self.tokenizer = tokenizer
        self.judge_by_token_id = judge_by_token_id
        self.prefix_space = ""
        if not 'llama' in self.tokenizer.name_or_path.lower():
            self.prefix_space = " "
            
        self.codes = {
            n: self.tokenizer.encode("{}{}".format(self.prefix_space, c), 
                                    add_special_tokens=False)[0]
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
        if debug:
            # critieria codes: {'start_mention_token': 426, 'end_mention_token': 500, 
            # 'start_entity_token': 518, 'end_entity_token': 4514}
            print(f"critieria codes: {self.codes}")
        self.end_entity_token = end_entity_token
        
        assert this_mention_str is not None, f"this_mention_str is None"
        self.start_length = start_length
        self.max_new_tokens = len(self.tokenizer.encode(this_mention_str, 
                                                        add_special_tokens=False))
        self.max_length = start_length + self.max_new_tokens
        self.ins_length = ins_length
        
        self.debug = debug
        
    def get_status(self, input_ids):
        c = [
            self.codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        assert self.ins_length is not None
        offset = 0 if self.ins_length is None else self.ins_length
        status = sum(e in c for e in input_ids[offset:]) % 4
        if self.debug:
            print(f"get_status in stopping critieria: {input_ids[offset:]}")
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
        
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # stop when 
        # a) no special tokens are found and exceed max_length;
        # b) last token is entity close token
        assert input_ids.shape[0] == 1 or len(input_ids.shape) == 2, "now only support batch_size=1"
        input_ids = input_ids[0]
        
        status = self.get_status(input_ids.tolist())
        # when outside a mention, stop when exceeding max_length
        if self.debug:
            print(f"status in stopping critieria: {status}; max_length is {self.max_length}; input_ids shape: {input_ids.shape}")
            # print(f"input_ids in stopping critieria: {input_ids.tolist()}")
        if status == "o":
            return input_ids.shape[-1] >= self.max_length
        # when inside a mention, never stop
        elif status == "m":
            # print(f"status in stopping m: never stop!")
            return False
        # when inside a entity, stop when entity close token is generated
        elif status == "e":
            last_token_id = input_ids[-1]
            if self.judge_by_token_id:
                return last_token_id == self.codes["end_entity_token"]
            else:
                return self.end_entity_token in self.tokenizer.decode(last_token_id, skip_special_tokens=True)

# stop when special mention_end token is generated
class BoundaryStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, boundary_end_token="}", judge_by_token_id=False):
        # `}` for mention end, `]` for entity end
        self.tokenizer = tokenizer
        self.judge_by_token_id = judge_by_token_id
        self.boundary_end_token = boundary_end_token
        self.prefix_space = ""
        if not 'llama' in self.tokenizer.name_or_path.lower():
            self.prefix_space = " "
        
        self.boundary_end_token_id = self.tokenizer.encode("{}{}".format(self.prefix_space, boundary_end_token), 
                                                          add_special_tokens=False)
        assert len(self.mboundary_end_token_id) == 1, "boundary_end_token_id should be a single token"
        self.boundary_end_token_id = self.boundary_end_token_id[0]
        
    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0][-1]
        if self.judge_by_token_id:
            return last_token_id == self.boundary_end_token_id
        else:
            last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)
            return self.boundary_end_token in last_token
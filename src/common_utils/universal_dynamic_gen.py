# on 05/24: this file implements an effieicent GenEL method
# with retrieved top-k entities + entity_to_mention_dict 
# + 32-token window
from typing import Optional, List, Dict, Tuple
from common_utils.retrieval_gen_utils import SelectedStoppingCriteria, get_retrieval_prefix_allowed_tokens_fn_mention
import re
import torch
import ahocorasick
from collections import defaultdict
from termcolor import cprint

class FuncWithDebug:
    def __init__(self, some_func, debug) -> None:
        self.some_func = some_func
        self.debug = debug
        # print(f"activate debug mode in prefix_func: {debug}")
        
    def __call__(self, *args, **kwargs):
        return self.some_func(*args, **kwargs)
    

def merge_m_e_pairs(mention_to_candidates_lst) -> List[Tuple[int, int, List[str]]]:
    """
    merge overlapped mentions:
    e.g. (6, 8, 'App'), (6, 10, 'Apple') -> (6, 10, ['App', 'Apple'])
    e.g.2 (67, 71, 'Orang'), (67, 72, 'Orange') -> (67, 72, ['Orang', 'Orange'])
    """
    merged_mentions = []
    mention_to_candidate_dict = dict()
    for start, end, mention, candidates_set in mention_to_candidates_lst:
        mention_to_candidate_dict[mention] = list(candidates_set)
        if merged_mentions and merged_mentions[-1][0] <= start <= merged_mentions[-1][1]:
            merged_mentions[-1][1] = max(merged_mentions[-1][1], end)
            merged_mentions[-1][2].append(mention)
        else:
            merged_mentions.append([start, end, [mention]])
    return merged_mentions, mention_to_candidate_dict

class UniversalRetrievalGenELWrapper:
    def __init__(self, model, tokenizer, task_instruction: str, debug=False, 
                 stopwords: List[str] = None, no_print=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.is_llama_tokenizer = 'llama' in tokenizer.name_or_path.lower()
        
        self.task_instruction = task_instruction
        self.ins_len = len(self.tokenizer.tokenize(task_instruction))
        
        self.generated_sent = ""
        
        self.debug = debug
        self.stopwords = set(stopwords)
        self.no_print = no_print
        
    def debug_print(self, caller, value, critical=False):
        if self.no_print:
            return
        if critical:
            cprint(f"{caller}: {value}", "red")
        if self.debug:
            cprint(f"{caller}: {value}", "yellow")

    def get_prompt(self, input_text, with_generated=False):
        ins = self.task_instruction
        if not self.is_llama_tokenizer:  # non-llama doc chunk will be prepended a space
            ins = ins.rstrip()
        input_text = self.preprocess_text(input_text).rstrip() + self.tokenizer.eos_token
        if not self.is_llama_tokenizer and \
            len(input_text) > 0 and input_text[0] != ' ':
                input_text = ' ' + input_text
                        
        if not with_generated:  # just begin
            return ins + input_text
        else:
            # update on 06/01: making sure boundary tokens in `self.generated_sent` are with spaces
            # generated_sent = re.sub(r"\].*?", "] ", self.generated_sent)
            return ins + input_text + self.generated_sent
    
    def reset_prompt(self):
        # run this once a doc_chunk is processed
        self.generated_sent = ""
        
    def get_span_mention_and_entities(self, doc_chunk, 
                                      entity_candidates: List[str],
                                      candidate_to_mention_dict: Dict[str, List[str]],
                                      missing_threshold=0.1,) -> List[Tuple[int, int, str, List[str]]]:
        automaton = ahocorasick.Automaton()
        mention_to_candidate_dict = defaultdict(set)
        
        key_error_count = 0
        for c in entity_candidates:
            try:
                for mention in candidate_to_mention_dict[c]:
                    # add mention filter here; comma-ending and short mentions waiting...
                    # update on 06/02: ignore surronding spaces
                    mention = mention.strip()
                    if mention and mention[0].isupper() and mention.lower() not in self.stopwords:
                        automaton.add_word(mention, mention)
                        mention_to_candidate_dict[mention].add(c)
                        
            except KeyError:
                key_error_count += 1
                continue
            
        if key_error_count / len(entity_candidates) > missing_threshold:
            self.debug_print("[get_span_mention_and_entities]", 
                             f"key_error_count: {key_error_count} / {len(entity_candidates)}", 
                             critical=True)
        
        ret = []
        automaton.make_automaton()
        # use doc_chunk to get possibly overlapping mentions and their entities
        haystack = doc_chunk
        for end_index, original_value in automaton.iter(haystack):
            start_index = end_index - len(original_value) + 1
            ret.append(
                (start_index, end_index, original_value, 
                list(mention_to_candidate_dict[original_value]))
            )
            assert haystack[start_index:start_index + len(original_value)] == original_value
        
        # assert start & end index is assending
        assert all([ret[i][0] <= ret[i][1] for i in range(len(ret))]), f"ret: {ret}"
        assert all([ret[i][1] <= ret[i + 1][1] for i in range(len(ret) - 1)]), f"ret: {ret}"
        # sort by start index
        ret = list(sorted(ret, key=lambda x: x[0]))
        assert all([ret[i][0] <= ret[i + 1][0] for i in range(len(ret) - 1)]), f"ret: {ret}"
        
        return ret
        
         
    @staticmethod
    def preprocess_text(text: str) -> str:
        # preprocess replace must not change index!
        return text.replace("\xa0", " ").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")").replace("\n", " ")
    
    @staticmethod
    def postprocess_text(sentences: List[str]) -> List[str]:  
        # before get_entity_span_finalized
        outputs = []
        for sent in sentences:
            sent = re.sub(r"{.*?", "{ ", sent)
            sent = re.sub(r"}.*?", "} ", sent)
            sent = re.sub(r"\].*?", "] ", sent)
            sent = re.sub(r"\[.*?", "[ ", sent)
            sent = re.sub(r"\s{2,}", " ", sent)
            sent = re.sub(r"\. \. \} \[ (.*?) \]", r". } [ \1 ] .", sent)
            sent = re.sub(r"\, \} \[ (.*?) \]", r" } [ \1 ] ,", sent)
            sent = re.sub(r"\; \} \[ (.*?) \]", r" } [ \1 ] ;", sent)
            sent = sent.replace("{ ", "{").replace(" } [ ", "}[").replace(" ]", "]")  # shorten the boundary
            outputs.append(sent)
        return outputs
    
    def compute_optimized_forward_times(self, merged_spans, mention_to_candidate_dict):
        # merged_spans: [[90, 99, ['Home Depot', 'Depot']], [118, 125, ['Nard', 'Nardelli']]], 
        # mention_to_candidate_dict: {'Home Depot': ['The Home Depot'], 'Depot': ['Depot', 'The Home Depot'], 'Nard': ['Nardò'], 'Nardelli': ['Nardelli', 'Robert Nardelli', 'Steve Nardelli', 'Francesco Nardelli']})
        sum_forward_passes = 0
        for span in merged_spans:
            sum_forward_passes += len(span[2])
            for possible_mention in span[2]:
                if len(mention_to_candidate_dict[possible_mention]) > 1:
                    sum_forward_passes += len(mention_to_candidate_dict[possible_mention])
                    
        return sum_forward_passes
    
    # note: we have to override model.generate; also sent splitting should be done outside, 
    # as GenEL in fact does not pose any restriction on chunking. 
    # on 05/24: retrieval generate does NOT support batch inference inherently
    def retrieval_generate(self, doc_chunk: str, num_beams=2, 
                           max_length=512, entity_candidate_lst: List[str] = None, 
                           candidate_to_mention_dict: Dict[str, List[str]] = None, expand_mention=False, 
                           get_optimized_forward_times=False, **model_generation_kwargs) -> str:
        """
        Caution:
            1. max_length should never be reached...
            2. possible_mention_lst should optionally expanded; we should test w/ and w/o mention_expansion
            3. top-k entities are better retrieved using wikipedia_id to title; 
        """
        self.reset_prompt()
            
        prompt_sent = self.get_prompt(doc_chunk, with_generated=False)
            
        self.debug_print("[prompt_sent]", prompt_sent)
            
        # begin to get top entities, mentions and merged spans
        spans = self.get_span_mention_and_entities(doc_chunk, entity_candidate_lst, 
                                                   candidate_to_mention_dict)

        self.debug_print("[spans]", spans)
        merged_spans, mention_to_candidate_dict = merge_m_e_pairs(spans)
        self.debug_print("[merged_spans]", merged_spans)
        
        if get_optimized_forward_times:
            optimized_forward_times = self.compute_optimized_forward_times(merged_spans, mention_to_candidate_dict)
        
        self.debug_print("[mention_to_candidate_dict]", mention_to_candidate_dict)
        # merged_spans: [[90, 99, ['Home Depot', 'Depot']], [118, 125, ['Nard', 'Nardelli']]], 
        # mention_to_candidate_dict: {'Home Depot': ['The Home Depot'], 'Depot': ['Depot', 'The Home Depot'], 'Nard': ['Nardò'], 'Nardelli': ['Nardelli', 'Robert Nardelli', 'Steve Nardelli', 'Francesco Nardelli']})
        
        # begin to iterating over merged_spans
        # first, copy until the first possible mention starts
        curr = 0
        last_word_len = 1
        # st & ed are both closed range
        for span_id, (st, ed, possible_mentions) in enumerate(merged_spans):
            # special treatment #1: for `Vic` and `Victoria|toria`
            offset = 1
            
            for off in range(2, last_word_len + 1):
                if doc_chunk[curr: curr + off] == self.generated_sent[-off:]:
                    self.debug_print(f"[special crop offset#{off}]", f"{doc_chunk[curr: curr + off]} v.s. {self.generated_sent[-off:]}", critical=True)
                    offset = off + 1
            
            self.generated_sent += doc_chunk[curr + offset - 1:st]
            
            fixed_prompt = self.get_prompt(doc_chunk, with_generated=False)
            prompt = self.get_prompt(doc_chunk, with_generated=True).rstrip()
            
            truc_sent = prompt.split(self.tokenizer.eos_token)[-1]  # current generated sent
            # assert the concat of fixed_prompt + truc_sent == prompt
            assert fixed_prompt + truc_sent == prompt
            
            self.debug_print("[prompt within merged_spans]", prompt)
            self.debug_print("[truc_sent in main func]", truc_sent)
            
            input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                prompt, add_special_tokens=False
            )
            # build prefix_allowed_tokens_fn on the fly
            prefix_allowed_tokens_fn = get_retrieval_prefix_allowed_tokens_fn_mention(
                self.tokenizer,
                possible_mentions, 
                # truc_sentence=truc_sent,
                truc_sentence=doc_chunk, 
                prompt_len=len(self.tokenizer.encode(fixed_prompt, 
                                                     add_special_tokens=False)) + 1,
                possible_mention_to_candidates_dict=mention_to_candidate_dict,
                debug=self.debug
            )
            prefix_allowed_tokens_fn = FuncWithDebug(prefix_allowed_tokens_fn, self.debug)
            
            response = self.model.generate(
                input_ids=torch.LongTensor([input_ids]).to(self.model.device), 
                num_return_sequences=1,
                num_beams=num_beams,
                stopping_criteria=[
                    SelectedStoppingCriteria(
                        tokenizer=self.tokenizer, 
                        judge_by_token_id=True, 
                        start_length=len(input_ids),
                        this_mention_str=doc_chunk[st: ed+1],
                        debug=self.debug,
                        ins_length=self.ins_len,
                    )
                ], 
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                max_length=max_length,
                **model_generation_kwargs
            )
            
            self.debug_print(f"[response from span#{span_id}]", response)
                
            raw_output = response.detach().cpu().numpy()
            
            # output should cover new generated m_e pair
            output: str = self.tokenizer.decode(raw_output[0], skip_special_tokens=False)
            # crop generated sent, and update self.generated_sent
            self.generated_sent = output.split(self.tokenizer.eos_token)[-1]  # #1: eos split [-1]
            # eos split -1 will be generated_sent, except the last span overlaps with original EOS
            # assert over multiple ways of getting generated_sent
            generated_sent = output[len(fixed_prompt) + len(self.tokenizer.eos_token):]  # #2: fixed_prompt split
            self.debug_print("[generated_sent compare]", f"`{generated_sent}`\n`{self.generated_sent}`")
            
            # generation complete for this span!
            curr = ed + 1  # update curr
            last_word_len = len(self.generated_sent.split()[-1])
            
            # special treatment #2: if selected mention is not the longest mention in possible mentions, 
            # we have to compensate the length, i.e. shorten curr for unselected mention 
            
            
        # remember to attach last part
        self.generated_sent += doc_chunk[curr:]
        self.debug_print("[after concat parts]", self.generated_sent)
        postprocessed_sent = self.postprocess_text([self.generated_sent])[0]
        self.debug_print("[after postprocess]", postprocessed_sent)
        
        # postprocessed sent should never contain <unk>
        if self.tokenizer.unk_token in postprocessed_sent:
            raise RuntimeError(f"postprocessed_sent contains <unk>: {postprocessed_sent}")
        
        if get_optimized_forward_times:
            return postprocessed_sent, optimized_forward_times
        return postprocessed_sent
    
if __name__ == '__main__':
    pass
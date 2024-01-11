from typing import Optional, List
import torch
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn  # trie class will be implicitly imported
# in future, we import from common_utils.el_prefix instead, as we need a universal prefix_fn over OPT/GPT/LLaMa/Bloom
from genre.entity_linking import Trie
import re

class FuncWithDebug:
    def __init__(self, some_func, debug) -> None:
        self.some_func = some_func
        self.debug = debug
        
    def __call__(self, *args, **kwargs):
        return self.some_func(*args, **kwargs)
    

class LlamaForConstrainedELWrapper:
    """
    More suitable for notebook debugging
    """
    def __init__(self, model) -> None:
        self.model = model
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        # preprocess replace must not change index!
        return text.replace("\xa0", " ").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")").replace("\n", " ")
    
    @staticmethod
    def postprocess_text(sentences: List[str]) -> List[str]:  # before get_entity_span_finalized
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
    
    def el_generate(self, instruction: str, sentences: List[str], tokenizer, max_length=512, num_beams=6, 
                    mention_trie: Optional[Trie] = None, mention_to_candidates_dict: Optional[dict] = None, 
                    skip_special_tokens=True, early_stopping=False, debug=False, return_raw=False, free_form_gen=False, 
                    **model_generation_kwargs) -> List[str]:
        # keep identical call signatures as fairscale model.sample()
        # Caution: When the input sequence is too long, we split the input into multiple chunks.
        # prefix_allowed_tokens_fn: lambda batch_id, sent: trie.get(sent.tolist()),
        
        # build prefix_fn first, as later we need to concat an instruction to the input
        for i in range(len(sentences)):
            sentences[i] = self.preprocess_text(sentences[i]).rstrip()  # preprocessed!
            sentences[i] += tokenizer.eos_token  # add eos_token to the end of each sentence
                
        if debug:
            print(f"after preprocessed: {sentences}")

        # now concat instruction to the input
        if isinstance(instruction, list):
            assert len(instruction) == len(sentences), f"len(instruction)={len(instruction)} != len(sentences)={len(sentences)}"
            prompt_sentences = [ins + sent for ins, sent in zip(instruction, sentences)]  # remove space since sent is guraranteed to start with space
        elif isinstance(instruction, str):
            prompt_sentences = [instruction + sent for sent in sentences]
        else:
            raise ValueError(f"Unknown type of instruction: {type(instruction)}")
        
        if debug:
            print(f"after concat instruction: {prompt_sentences}")
            
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            tokenizer, 
            sentences,  # instruction not included
            prompt_sentences,  # include instruction + input
            mention_trie=mention_trie,
            mention_to_candidates_dict=mention_to_candidates_dict,
            debug=debug,
        )
        
        prefix_allowed_tokens_fn = FuncWithDebug(prefix_allowed_tokens_fn, debug)
        
        outputs = self.model.generate(
            input_ids=tokenizer(prompt_sentences, return_tensors="pt").input_ids.to(self.model.device),
            # in training, padding was added to left, so currently we do not support batch inference.
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=early_stopping,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if not free_form_gen else None,
            **model_generation_kwargs,
        )
        if debug:
            print(f"see outputs: {outputs}")
            
        # if any output does not end with EOS, the generation is incomplete.
        if return_raw:
            return outputs
        
        raw_outputs = outputs.detach().cpu().numpy()
        
        outputs: List[str] = tokenizer.batch_decode(raw_outputs, 
                                                    skip_special_tokens=skip_special_tokens)  # constrained output
        # strip instruction + input parts 
        COMPENSATE_CHARS = 3
        assert all(len(prompt_sentences[i]) - COMPENSATE_CHARS > 0 for i in range(len(outputs))), f"Stripping Error!"
        outputs = [out[len(prompt_sentences[i]) - COMPENSATE_CHARS:] for i, out in enumerate(outputs)]
        # WARNING: if skipped, should compensate for the extra </s> token``
        
        if debug:
            print(f"after decode: {outputs}")
            
        if any(not out[-1] == tokenizer.eos_token_id for out in raw_outputs):
            raise RuntimeError(f"Generation not complete for this batch! Recommending to regenerate using small seq_len.")
        # post-process for get_entity_span_finalized; splitting and computing metrics left to eval_server
        outputs = self.postprocess_text(outputs)
        if debug:
            print(f"after postprocess: {outputs}")
        
        return outputs
    
    
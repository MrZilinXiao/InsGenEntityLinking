# Update on 05/10: modification complete except for COMPENSATE_TOKEN_LEN
from typing import Optional, List
# from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn  # trie class will be implicitly imported
# in future, we import from common_utils.el_prefix instead, as we need a universal prefix_fn over OPT/GPT/LLaMa/Bloom
from genre.entity_linking import Trie
# from transformers import GenerationMixin  # noqa, for step-in reference only
from .el_prefix import get_prefix_allowed_tokens_fn_universal as get_prefix_allowed_tokens_fn
import re
import torch

class FuncWithDebug:
    def __init__(self, some_func, debug) -> None:
        self.some_func = some_func
        self.debug = debug
        # print(f"activate debug mode in prefix_func: {debug}")
        
    def __call__(self, *args, **kwargs):
        return self.some_func(*args, **kwargs)
    

class UniversalConstrainedELWrapper:
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
                    skip_special_tokens=False, early_stopping=False, debug=False, return_raw=False, free_form_gen=False, 
                    **model_generation_kwargs) -> List[str]:
        # keep identical call signatures as fairscale model.sample()
        # Caution: When the input sequence is too long, we split the input into multiple chunks.
        # prefix_allowed_tokens_fn: lambda batch_id, sent: trie.get(sent.tolist()),
        
        # build prefix_fn first, as later we need to concat an instruction to the input
        for i in range(len(sentences)):
            sentences[i] = self.preprocess_text(sentences[i]).rstrip()  # preprocessed!
            sentences[i] += tokenizer.eos_token  # add eos_token to the end of each sentence
            # update on 05/15: non-llama model needs prepending a space
            if 'llama' not in tokenizer.name_or_path.lower() and \
                len(sentences[i]) > 0 and sentences[i][0] != ' ':
                    sentences[i] = ' ' + sentences[i]
                
        if debug:
            print(f"after preprocessed: {sentences}")

        # now concat instruction to the input
        if isinstance(instruction, list):
            assert len(instruction) == len(sentences), f"len(instruction)={len(instruction)} != len(sentences)={len(sentences)}"
            prompt_sentences = [ins + sent for ins, sent in zip(instruction, sentences)]  # remove space since sent is guraranteed to start with space
        elif isinstance(instruction, str):
            assert instruction.endswith(" "), f"Instruction must end with space!"
            if 'llama' not in tokenizer.name_or_path.lower():
                instruction = instruction.rstrip()  # opt-style model, we offload the space in original sentences
            prompt_sentences = [instruction + sent for sent in sentences]  # instruction must be appended with space
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
        
        # added for universal usage
        assert len(prompt_sentences) == 1, f"Currently we do not support batch inference for universal generation!"
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(
            prompt_sentences[0], 
            add_special_tokens=False
        )
        
        outputs = self.model.generate(
            input_ids=torch.LongTensor([input_ids]).to(self.model.device),
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
        
        if any(not out[-1] == tokenizer.eos_token_id for out in raw_outputs):
            raise RuntimeError(f"Generation not complete for this batch! Recommending to regenerate using small seq_len.")
        
        outputs: List[str] = tokenizer.batch_decode(raw_outputs, 
                                                    skip_special_tokens=skip_special_tokens)  # constrained output
        # output sequence should be between two/three EOS
        if tokenizer.bos_token_id == tokenizer.eos_token_id:  # opt / gpt
            assert all(output.count(tokenizer.eos_token) == 3 for output in outputs), f"Num of EOS not equal to 3! {outputs}"
        else:  # llama / bloom
            assert all(output.count(tokenizer.eos_token) == 2 for output in outputs), f"Num of EOS not equal to 2! {outputs}"
        
        outputs = [output.split(tokenizer.eos_token)[-2] for output in outputs]
        
        # strip instruction + input parts 
        # TODO: compensate_chars may differ between models, relevant to tokenizer.eos_token char length
        # COMPENSATE_CHARS = 3 if 'llama' in tokenizer.name_or_path.lower() else 4  # 4 works for OPT; 3 works for llama
        # if compensate_char_num:
        #     COMPENSATE_CHARS = compensate_char_num
        # assert all(len(prompt_sentences[i]) - COMPENSATE_CHARS > 0 for i in range(len(outputs))), f"Stripping Error!"
        # outputs = [out[len(prompt_sentences[i]) - COMPENSATE_CHARS:] for i, out in enumerate(outputs)]
        # WARNING: if skipped, should compensate for the extra </s> token``
        
        if debug:
            print(f"after decode: {outputs}")
            
        # post-process for get_entity_span_finalized; splitting and computing metrics left to eval_server
        outputs = self.postprocess_text(outputs)
        if debug:
            print(f"after postprocess: {outputs}")
        
        return outputs
    
    
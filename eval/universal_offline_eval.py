# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python EL_universal/universal_offline_eval.py --watch_path ./opt_1.3b_EL_0.5 --checkpoint_name . --tags fix_punc
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src/ python EL_universal/universal_offline_eval.py --watch_path ./opt-350m_EL_0.5 --checkpoint_name . --tags fix_punc
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src/ python EL_universal/universal_offline_eval.py --watch_path ./opt-2.7b_EL_0.5 --checkpoint_name . --tags fix_punc
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python EL_universal/universal_offline_eval.py --watch_path ./opt-6.7b_EL_0.2 --checkpoint_name . --tags fix_punc
import argparse
from typing import List, Union
import pickle
import requests
import spacy
from common_utils.universal_constrained_gen import UniversalConstrainedELWrapper
from tqdm import tqdm
from itertools import product
from glob import glob
import time
from peft import PeftModel
from datetime import datetime
from packaging import version
import torch
# under PYTHONPATH src/
import sys
import os
import re
import json
os.environ['PYTHONPATH'] = "/home/v-zilinxiao/code/transformers/src"
sys.path.insert(0, "/home/v-zilinxiao/code/transformers/src")
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(f"transformers version: {transformers.__version__}")  # should end with dev0
assert transformers.__version__.endswith("dev0")

from common_utils.split import split_text_to_parts, split_sentence

# CONTROL DATA LOADING FLAG
LOAD_DATA = True

GET_OLD_DATA = False

if GET_OLD_DATA:
    print("WARNING: Using deprecated data!!!")
    
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=None)  # use k50 for preliminary evaluation
parser.add_argument("--exclude_datasets", nargs="+", default=None)

parser.add_argument("--check_freq", type=int, default=10)
parser.add_argument("--watch_path", nargs="+", required=True)
parser.add_argument("--checkpoint_name", nargs="+", default=None)
parser.add_argument("--ignored_path", nargs="+", default=None)
parser.add_argument("--beam_size", nargs="+", default=None)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--split_size", type=int, default=175)
# update on 05/23: what if we just split by sentence? will MSNBC benefit?
parser.add_argument("--split_by_sent", action="store_true")

parser.add_argument("--tags", nargs="+", default=None)
parser.add_argument("--eval_subfolder", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

if args.debug:
    LOAD_DATA = False

# CONSTANTS: KNOW WHAT YOU ARE DOING
MAX_LENGTH = 512    # finetuned LLaMA was trained with max_length=512
INSTRUCTIONS = [
    "For each source provided, write an output that includes the mentions that linked to the appropriate entity identifier. Use brace brackets { } to denote a mention, and box brackets [ ] to denote an linked entity: ",
    "Make sure to include the original text of the source in the output, with the relevant mention words linked to the entity identifier. Use brace brackets `{` `}` to denote a mention, and box brackets `[` `]` to denote an linked entity: ",
    "The output should be formatted as a target text, with the original source text included, and the relevant mention words linked to the appropriate entity title. Use brace brackets `{` `}` to denote a mention, and box brackets `[` `]` to denote an linked entity: ",
]
BEAMS_RANGE = [2]  # combination #3
BASE_MODEL_PATH = "/home/v-zilinxiao/code/transformers/llama_hf_7B"  # original llama, useful when using lora
DUMMY_LORA_CONFIG_PATH = '/home/v-zilinxiao/code/transformers/EL_scripts/dummy_lora_ckpt/adapter_config.json'

if LOAD_DATA:
    MENTION_TRIE_MAPPING = {  # combination #1
        'evalonly': None,
        'full': None,   # full mention trie will lead to less precision usually
    }

    print(f'Loading mention tries...')
    # with open('/home/v-zilinxiao/code/transformers/genre_additional_data/0427_llama_mention_trie_el_eval_only.pkl', 'rb') as f:
    #     MENTION_TRIE_MAPPING['evalonly'] = pickle.load(f)
        
    if 'llama' in args.watch_path[0].lower():
        with open('/home/v-zilinxiao/code/transformers/genre_additional_data/0501_llama_mention_trie_full.pkl', 'rb') as f:
            MENTION_TRIE_MAPPING['full'] = pickle.load(f)
    elif 'opt' in args.watch_path[0].lower():
        with open('/home/v-zilinxiao/code/transformers/genre_additional_data/0511_opt-125m_mention_trie_full.pkl', 'rb') as f:
            MENTION_TRIE_MAPPING['full'] = pickle.load(f)
    else:
        raise NotImplementedError(f"Unknown model type: {args.watch_path[0]}")

    # load necessary dependencies
    from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
    import pickle
    from typing import List
    import urllib.parse

    print(f"Loading Necessary Data...")
    with open('/home/v-zilinxiao/code/transformers/src/genre/data/0501_mention_to_candidates.pkl', 'rb') as f:
        # covers full candidates; just do not select those invalid mentions by mention_trie
        mention_to_candidates_dict = pickle.load(f)
    
    if not GET_OLD_DATA:
        with open('/home/v-zilinxiao/code/transformers/src/genre/data/elevant/0220_link_redirects.pkl', 'rb') as f:  # used in transform_predictions
            link_redirects = pickle.load(f)  # type(link_redirects): dict entity -> entity
    else:
        with open('/home/v-zilinxiao/code/transformers/src/genre/data/elevant/link_redirects.pkl', 'rb') as f:
            link_redirects = pickle.load(f)  # type(link_redirects): dict entity -> entity

    def get_mapping():  # from wikipedia title to qid; used in transform_predictions
        with open('/home/v-zilinxiao/data/elevant_data/wikidata_mappings/wikipedia_name_to_qid.pkl', 'rb') as f:
            mapping = pickle.load(f)
            
        # reload mapping by decoding
        mapping_keys = list(mapping.keys())
        for k in tqdm(mapping_keys):
            if isinstance(k, bytes):
                _k = k.decode()
                mapping[_k] = mapping[k].decode() if isinstance(mapping[k], bytes) else mapping[k]
                del mapping[k]
            elif isinstance(mapping[k], bytes):
                mapping[k] = mapping[k].decode()
        return mapping
    
    def get_old_mapping():
        prefix = "https://en.wikipedia.org/wiki/"
        mapping = {}
        for line in tqdm(open("/home/v-zilinxiao/data/code/transformers/src/genre/data/elevant/qid_to_wikipedia_url.tsv")):
            line = line[:-1]
            vals = line.split("\t")
            qid = vals[0]
            wikipedia_title = urllib.parse.unquote(vals[1][len(prefix):]).replace("_", " ")
            mapping[wikipedia_title] = qid
        return mapping
        
    mapping = get_mapping() if not GET_OLD_DATA else get_old_mapping()  # type(mapping): title -> qid


def delete_model(model):
    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    
def add_space_before_punctuation(text):
    text = re.sub(r'(?<!\s)([.,!?;:"`~@#\$%\^&\*\(\)<>])', r' \1', text)
    # hot patch 06/02: replace strange U .S . and U .K . with U.S. and U.K.
    text = text.replace("U .S .", "U.S.").replace("U .K .", "U.K.")
    return text

# eval dataset parser
dataset_base_path = '/home/v-zilinxiao/code/elevant/benchmarks/'

dataset_paths = {  # combination #2
    'k50': dataset_base_path + 'kore50.benchmark.jsonl',
    'msnbc': dataset_base_path + 'msnbc.benchmark.jsonl',
    'der': dataset_base_path + 'derczynski.benchmark.jsonl',
    'r500': dataset_base_path + 'rss-500.benchmark.jsonl',  # in benchmark but not recommended
    'r128': dataset_base_path + 'reuters-128.benchmark.jsonl',
    'oke15': dataset_base_path + 'oke-2015-eval.benchmark.jsonl',
    'oke16': dataset_base_path + 'oke-2016-eval.benchmark.jsonl',
    'aida_test': dataset_base_path + 'aida-clean-test.benchmark.jsonl',
}

abbr_dataset_to_full_name = {
    'aida_test': 'aida-clean-test',
    'k50': 'kore50',
    'msnbc': 'msnbc',
    'der': 'derczynski',
    'r500': 'rss-500',
    'r128': 'reuters-128', 
    'oke15': 'oke-2015-eval', 
    'oke16': 'oke-2016-eval'
}

# util functions, do not touch

def parse_jsonl(jsonl_file):
    with open(jsonl_file, "r") as f:
        for line in f:
            yield json.loads(line)


def parse_dataset(jsonl_file):
    for data in parse_jsonl(jsonl_file):  # for each line
        # data: {"id": 0, "title": "http://aksw.org/N3/RSS-500/0#char=0,150", "text": "The U.S. Patent Office allows genes to be patented as soon as someone isolates the DNA by removing it from the cell , says ACLU attorney Sandra Park .",
        # "evaluation_span": [0, 150], "labels": [{"id": 0, "span": [123, 127], "entity_id": "Q21637", "name": "American Civil Liberties Union", "parent": null, "children": [], "optional": false, "type": "Q43229"},
        # {"id": 1, "span": [137, 148], "entity_id": "Unknown", "name": "UnknownNoMapping", "parent": null, "children": [], "optional": false, "type": "OTHER"}]}
        text = data.pop('text')
        if 'evaluation_span' in data:
            eval_span = data['evaluation_span']
        else:
            eval_span = [0, len(text)]
        text = text[eval_span[0]: eval_span[1]]
        labels = data.pop('labels')
        # for label in labels:
        #     if any(label[k] for k in ['parent', 'children', 'optional']):
        #         print(f"WARNING: interesting at {data['id']} of {jsonl_file}: {[label[k] for k in ['parent', 'children', 'optional']]}")
        # should not touch original!
        yield {
            'text': text,  # need this only for later eval
            'labels': [{'mention_text': text[label['span'][0]: label['span'][1]],
                       **label}
                       for label in labels],
            **data  # unused params goes back
        }

# our adopted version
def llama_get_entity_spans_to_elevant_format(input_, output_, redirections=None, mapping=None):
    # output unnormalized entity span and identifier; span is denoted with [start, end]
    # recommend to provide redirects and mapping
    if redirections is None or mapping is None:
        print("WARNING: no redirections or mapping provided!")
    input_ = input_.replace("\xa0", " ") + "  -"
    output_ = output_.replace("\xa0", " ") + "  -"
    entities = []
    status = "o"
    i = 0
    j = 0
    complete = True
    # update on 04/17: allow return partially identified entities if incomplete generation
    try:
        while j < len(output_) and i < len(input_):
            if status == "o":
                if input_[i] == output_[j] or (
                    output_[j] in "()" and input_[i] in "[]{}"
                ):
                    i += 1
                    j += 1
                elif output_[j] == " ":
                    j += 1
                elif input_[i] == " ":
                    i += 1
                elif output_[j] == "{":
                    entities.append([i, 0, ""])
                    j += 1
                    status = "m"
                else:
                    raise RuntimeError
            elif status == "m":
                if input_[i] == output_[j]:
                    i += 1
                    j += 1
                    entities[-1][1] += 1
                elif output_[j] == " ":
                    j += 1
                elif input_[i] == " ":
                    i += 1
                elif output_[j] == "}":
                    j += 1
                    status = "e"
                else:
                    raise RuntimeError
            elif status == "e":
                if output_[j] == "[":
                    j += 1
                elif output_[j] != "]":
                    entities[-1][2] += output_[j]
                    j += 1
                elif output_[j] == "]":
                    # entities[-1][2] = entities[-1][2].replace(" ", "_")  # do not normalize
                    if len(entities[-1][2]) <= 1:
                        del entities[-1]
                    elif entities[-1][2] == "NIL":  # do not link to NIL
                        del entities[-1]

                    elif mapping is not None and entities[-1][2] in mapping:
                        entities[-1][2] = mapping[entities[-1][2]]  # direct to QID

                    elif redirections is not None and entities[-1][2] in redirections:
                        entities[-1][2] = redirections[entities[-1][2]]
                        if mapping is not None and entities[-1][2] in mapping:
                            entities[-1][2] = mapping[entities[-1][2]]  # redirected then to QID
                    # if all missed, keep the entity identifier
                    if len(entities) > 0:
                        entities[-1] = tuple(entities[-1])
                    status = "o"
                    j += 1
                else:
                    raise RuntimeError
    except RuntimeError:
        complete = False
        print("WARNING: incomplete generation, return partially correct entities...")

    # convert gerbil format to elevant, turn len to end
    entities = [(start, start + len_, entity) for start, len_, entity in entities]
    return entities, complete


def create_label_json(begin, end, qid, model_name="GENRE"):
    return {
        "span": [begin, end],
        "recognized_by": model_name,
        "id": qid,
        "linked_by": model_name,
        "candidates": [qid]
    }


# scanning utils; do not need multi-thread since one GPU is used for eval
class CheckpointWatcher:
    def __init__(self, watch_path: Union[str, List[str]],
                 check_freq: int = 10,
                 ignored_paths: List[str] = None, 
                 checkpoint_name: List[str] = None) -> None:
        self.watch_path = watch_path   # the split[-2] part will be used as exp identifier, so make sure it is unique
        if isinstance(watch_path, str):
            self.watch_path = [watch_path]
        self.checkpoint_name = checkpoint_name
        
        self.check_freq = check_freq
        self.ignored_paths = ignored_paths   # skip some checkpoints

        self.seen_folders = set()
        self.spacy_model = spacy.load("en_core_web_sm")
        
        self.lora_base_model = None

    def get_folders(self):
        # glob returns paths as what you give it
        # print(f"DEBUG: args: {args}, checkpoint_name: {self.checkpoint_name}")
        if self.checkpoint_name is None:
            for watch_path in self.watch_path:
                for abs_path in glob(os.path.join(watch_path, "checkpoint-*")):
                    if abs_path in self.seen_folders:  # exact match will be skipped
                        continue
                    if self.ignored_paths is not None:
                        if any(ignored_path in abs_path for ignored_path in self.ignored_paths):
                            # partial match (usually on exp_name/checkpoint-xxx) will be skipped
                            continue
                    yield abs_path
        else:
            assert len(self.watch_path) == 1
            for checkpoint_name in self.checkpoint_name:
                if checkpoint_name == '.':
                    if self.watch_path[0] not in self.seen_folders:
                        yield self.watch_path[0]
                    return   # do not repeatly eval the same checkpoint
                for abs_path in glob(os.path.join(self.watch_path[0], checkpoint_name)):
                    if abs_path in self.seen_folders:
                        continue
                    # when checkpoint_name is given, no need to check ignored_paths
                    yield abs_path

    def convert_from_zero3(self, ckpt_path):
        if not os.path.exists(os.path.join(ckpt_path, "zero_to_fp32.py")):
            assert os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin"))
            return  # no need for conversion
        print(f"converting {ckpt_path} from zero3 to fp32...")
        # step into the folder and run the conversion script
        if 'lora' in ckpt_path.lower():
            # lora ckpt, target_filename should be `adapter_model.bin`, and dump a adapter_config.json in this path
            os.system(f"cp {DUMMY_LORA_CONFIG_PATH} {ckpt_path}")
            target_filename = "adapter_model.bin"
        else:
            target_filename = "pytorch_model.bin"
        
        # if the target file already exists, skip
        if os.path.exists(os.path.join(ckpt_path, target_filename)):
            print(f"target file {ckpt_path}/{target_filename} already exists, skip conversion!")
            return
        os.system(f"cd {ckpt_path} && python zero_to_fp32.py . {target_filename}")

    def load_genel_model(self, ckpt_path):
        device_map = {"": 0} if version.parse(torch.__version__) > version.parse("1.10") else None
        # device_map = None
        # print(f"Set device map to {device_map}!")
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
        )
        if 'lora' in ckpt_path.lower():
            device_map = None   # lora base model is not compatible with device map
            print(f"loading lora from {ckpt_path}...")
            if self.lora_base_model is None:
                # first time using lora, load the base model
                print(f"loading lora base model...")
                self.lora_base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_PATH,
                    torch_dtype=torch.float16,
                    load_in_8bit=False,
                    cache_dir="cache",
                    device_map=device_map,
                )
                print(f"loading lora base model done!")
            model = PeftModel.from_pretrained(self.lora_base_model, ckpt_path)
        else:
            print(f"loading trained checkpoint...")
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float16,
                load_in_8bit=False,
                cache_dir="cache",
                device_map=device_map,
            )

        if device_map is None:
            model = model.cuda()
        model = model.eval()

        return model, tokenizer
    
    def generate_by_sent(self, wrapper, input_text, tokenizer, trie, real_max_length, beam_size, debug) -> str:
        sents = split_sentence(self.spacy_model, input_text)
        try:
            output_text = []
            for part in sents:
                outputs = wrapper.el_generate(
                    instruction=INSTRUCTIONS[0],  # now use the first instruction
                    sentences=[part],
                    tokenizer=tokenizer,
                    mention_trie=MENTION_TRIE_MAPPING[trie],
                    max_length=real_max_length,
                    early_stopping='never',
                    num_beams=beam_size,
                    mention_to_candidates_dict=mention_to_candidates_dict,
                    debug=debug
                )
                output_text.append(outputs[0])
            output_text = " ".join(output_text)  # final prediction
        except RuntimeError as e:
            if str(e).startswith("Generation not complete"):
                # not okay even by sentence, we have to consider crop in sentence
                print("Generation not complete even by sentence, give up, providing partial output...")
                output_text = " ".join(output_text)
        
        return output_text

    def get_pred_jsonl(self, model, model_name, tokenizer, dataset_path, beam_size, trie: str,
                       split_size=175, split_type='average',
                       max_length=512, debug=False) -> List[dict]:
        output = []
        wrapper = UniversalConstrainedELWrapper(model)
        for sample in tqdm(parse_dataset(dataset_path), desc=f"evaluating {dataset_path}"):
            input_text = UniversalConstrainedELWrapper.preprocess_text(sample["text"])
            # preprocess first, this does not change any indexes
            
            # added 04/20: overwrite original sample['text']
            # sample["text"] = input_text
            labels = []
            # get split parts
            split_retry_count = 10
            # restore the original split size and max length if retry is triggered
            real_split_size = split_size
            real_max_length = max_length
            split_by_punc = False   # if True, will split more aggressively
            split_by_space = dataset_path == dataset_paths["aida_test"]
            while split_retry_count >= 0:
                try:
                    split_parts, num_sents = split_text_to_parts(
                        self.spacy_model,
                        input_text,
                        tokenizer,
                        max_allowed_tokens_each_part=real_split_size,
                        split_type=split_type,
                        split_by_punc=split_by_punc,
                        split_by_space=split_by_space
                    )
                    sample['split_parts'] = split_parts  # record how the text is split last time
                    # have to make sure split parts are identical with the original text
                    # assert " ".join(split_parts) == input_text
                    output_text = []
                    for part in split_parts:
                        part = add_space_before_punctuation(part)
                        outputs = wrapper.el_generate(
                            instruction=INSTRUCTIONS[0],  # now use the first instruction
                            sentences=[part],
                            tokenizer=tokenizer,
                            mention_trie=MENTION_TRIE_MAPPING[trie],
                            max_length=real_max_length,
                            num_beams=beam_size,
                            early_stopping='never',
                            mention_to_candidates_dict=mention_to_candidates_dict,
                            debug=debug
                        )
                        output_text.append(outputs[0])
                    output_text = " ".join(output_text)  # final prediction
                    break
                except RuntimeError as e:  # incomplete generation for any part; raised in `el_generate`
                    if str(e).startswith("Generation not complete"):  # happens when: small beam_size and too long context
                        split_retry_count -= 1
                        real_split_size = int(real_split_size * 0.8)
                        print(f"Regenerate split parts... Retry count: {split_retry_count}/10; split_size: {real_split_size}; num_sents: {num_sents}")
                        if split_retry_count < 8:  # previous 2 runs use split_by_sent
                            split_by_punc = True
                        if split_retry_count < 5 and num_sents == len(split_parts):
                            # no where to split; just break
                            print(f"Cannot split further, break...")
                            break
                        # real_max_length = int(real_max_length * 1.1)
                    else:
                        raise e
            if isinstance(output_text, list):
                output_text = " ".join(output_text)  # corner case when the first sentence triggered incomplete generation
            predicted, complete = llama_get_entity_spans_to_elevant_format(input_text, output_text, link_redirects, mapping)
            # RuntimeError raise because:
            # 1. imcomplete generation caused unmatched input & output;
            # 2. too small beam size caused all fall backs failed...
            if not complete:
                print(f"Incomplete Generation\ninput _text: {input_text};\noutput_text: {output_text}\n----------")
                # in this extre case, we do not consider any cross-sentence context, just submit each sentence.
                output_text = self.generate_by_sent(wrapper, input_text, tokenizer, trie, real_max_length, beam_size, debug)
                predicted, complete = llama_get_entity_spans_to_elevant_format(input_text, output_text, link_redirects, mapping)
                sample['failed_generation'] = f"Incomplete Generation after split_part trials"
                if not complete:
                    print(f"Incomplete Generation after generate_by_sent\ninput _text: {input_text};\noutput_text: {output_text}\n----------")
                    sample['failed_generation'] = f'Incomplete Generation after generate_by_sent'
            
            for begin, end, qid in predicted:
                labels.append(create_label_json(begin, end, qid, model_name))
            sample['entity_mentions'] = labels
            sample['generated_text'] = output_text

            output.append(sample)
            if debug:
                print(sample)
        return output
    
    def get_final_steps(self, ckpt_path):
        if os.path.exists(os.path.join(ckpt_path, "latest")):
            return open(os.path.join(ckpt_path, "latest")).readline().strip()
        # read from trainer_state.json if zero3 is not enabled
        assert os.path.exists(os.path.join(ckpt_path, "trainer_state.json")), f"Cannot find trainer_state.json in {ckpt_path}"
        with open(os.path.join(ckpt_path, "trainer_state.json")) as f:
            trainer_state = json.load(f)
        return f"global_step{trainer_state['global_step']}"

    def watch_forever(self, dataset_keys: List[str] = None, exclude_datasets: List[str] = None, beam_size: List[int] = None,
                      debug: bool = False, tags: List[str] = None, max_length: int = 512, split_size=175, 
                      eval_subfolder: str = None, split_by_sent: bool = False):
        print(f"watching {self.watch_path}...")
        if dataset_keys is None:
            dataset_keys = list(dataset_paths.keys())
        else:
            assert all(key in dataset_paths for key in dataset_keys), f"Unsupported dataset keys: {dataset_keys}"
        if exclude_datasets is not None:
            dataset_keys = [key for key in dataset_keys if key not in exclude_datasets]
        exp_suffix = ""
        if tags:
            exp_suffix = f"_{'-'.join(tags)}"
        no_more_times = 1
        while True:
            for folder in self.get_folders():  # for each checkpoint...
                print(f"Checking {folder}...")
                # periodically scan the checkpoint directory, run zero-3 conversion if possible.
                # load in model and submit evaluation against benchmarks, save each results to elevant file and run elevant collections
                # 1. convert from zero3 if possible
                self.convert_from_zero3(folder)  # wait for test...
                # 2. load the model & tokenizer
                model, tokenizer = self.load_genel_model(folder)
                # 3. run evaluation and save results; note the combination of params!
                # things to combine a) beam size, b) dataset, c) mention_trie (eval_only or full) | full is more reasonable
                
                # OVERRIDE BEAM SIZE
                beams = BEAMS_RANGE if beam_size is None else beam_size
                if type(beams) == int:
                    beams = [beams]
                beams = [int(beam) for beam in beams]
                split_type = "average"
                if split_by_sent:
                    split_type = "sent"
                for beam_size, dataset, trie, split_type in product(beams,
                                                                    dataset_keys,
                                                                    ["full"],
                                                                    [split_type]):  # 1024 split tends to be unlimited
                    if dataset in ['k50', 'der', 'r500'] and split_type == 'clip':
                        # k50, der, r500 is too small to be clipped
                        continue
                    dataset_path = dataset_paths[dataset]
                    elevant_dataset_name = abbr_dataset_to_full_name[dataset]
                    # model_name is the -2 element of path splits
                    if not "checkpoint" in folder.split("/")[-1]:  # if test final model, checkpoint name is empty
                        final_steps = self.get_final_steps(folder)
                        model_name = folder.split("/")[-1] + '-' + final_steps + \
                            f"beam{beam_size}-trie{trie}-split{split_type}" + exp_suffix
                    else:
                        model_name = folder.split("/")[-2] + '-' + folder.split("/")[-1] + \
                            f"beam{beam_size}-trie{trie}-split{split_type}" + exp_suffix  # with checkpoint number; dataset will be included later
                        
                        
                    pred_list = self.get_pred_jsonl(model, model_name, tokenizer,
                                                    dataset_path, beam_size,
                                                    trie, split_type=split_type,
                                                    max_length=max_length,
                                                    split_size=split_size,
                                                    )
                    # submit to elevant and wait for results...
                    response = None
                    while response is None:
                        print(f"submitting to elevant for {dataset} {model_name} {beam_size} {trie} {split_type}...")
                        try:
                            response = requests.post(
                                "http://localhost:8001/elevant_processing",
                                json={'dataset': elevant_dataset_name,
                                    'model': model_name,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                                    'predictions': pred_list,
                                    'eval_subfolder': eval_subfolder,
                                }, 
                            )
                            response = response.json()
                        except (requests.exceptions.JSONDecodeError, requests.exceptions.ConnectionError) as e:
                            response = None
                            print(f"Error: {e}; retrying after 5 seconds...")
                            time.sleep(5)
                        # receive response (evaluator.get_results_dict())
                    # save response in place
                    with open(f"{folder}/elevant_results_" + f"beam{beam_size}-trie{trie}.json", "w") as f:
                        json.dump(response, f, indent=2)

                self.seen_folders.add(folder)  # add to seen folders, evaluate the next
                # clean up the model
                delete_model(model)

            print(f"\r({no_more_times}) No more checkpoints to evaluate, sleeping for {self.check_freq} seconds...", end="", flush=True)
            no_more_times += 1
            time.sleep(self.check_freq)


if __name__ == "__main__":
    watcher = CheckpointWatcher(args.watch_path,
                                check_freq=args.check_freq,
                                ignored_paths=args.ignored_path, 
                                checkpoint_name=args.checkpoint_name,
                                )
    try:
        print("Starting to watch...")
        watcher.watch_forever(dataset_keys=args.datasets, exclude_datasets=args.exclude_datasets, 
                              beam_size=args.beam_size, tags=args.tags, debug=args.debug, split_size=args.split_size,
                              max_length=args.max_length, eval_subfolder=args.eval_subfolder, 
                              split_by_sent=args.split_by_sent,)
    except KeyboardInterrupt:
        print("Exiting...")

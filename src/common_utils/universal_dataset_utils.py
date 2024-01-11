# update on 05/10: both hf Trainer and DeepspeedChat can use this universal datasetw utils
import transformers
if not transformers.__version__.endswith("dev0"):
    raise ImportError("You are not using the latest version of transformers.")
import copy
from typing import Optional, Dict, Sequence
import logging
import copy
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
from torch.utils.data import Dataset, ConcatDataset
import random
import json
import os
from transformers import default_data_collator
import math

DEFAULT_INSTRUCTION = "For each source provided, write an output that includes the mentions that linked to the appropriate entity identifier. Use brace brackets { } to denote a mention, and box brackets [ ] to denote an linked entity: "

IGNORE_INDEX = -100

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
        
def tokenize_universal(src, tgt, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    train_sample = src + tgt
    # maually add bos here, eos will be added in function call
    sample_tokenized = [tokenizer.bos_token_id] + tokenizer.encode(train_sample, add_special_tokens=False)
    src_tokenized = [tokenizer.bos_token_id] + tokenizer.encode(src, add_special_tokens=False)
    labels = copy.deepcopy(sample_tokenized)
    
    src_len = len(src_tokenized) - 1  # -1 for src eos explosure
    labels[:src_len] = src_len * [IGNORE_INDEX]
    return dict(input_ids=sample_tokenized, labels=labels)


def append_space_if_not_ended_with_space(string: str) -> str:
    """Append a space if the string does not end with a space."""
    if string.endswith(" "):
        return string
    else:
        return string + " "
    
    
class UniversalJsonlDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer: transformers.PreTrainedTokenizer = None, 
                 instruction_file=None, empty_instruction=False) -> None:
        self.tokenizer = tokenizer
        with open(jsonl_file, "r") as f:
            self.data: List[dict] = [json.loads(line) for line in f]  
            # TODO: use Serilized Toolbox to save memory? only when we publish the code
        logging.warning(f"Loaded {len(self.data)} examples...")
        
        self.instructions = []
        if instruction_file:
            with open(instruction_file, "r") as f:
                self.instructions = [line.rstrip('\n') for line in f.readlines()]
            logging.warning(f"Loaded {len(self.instructions)} instructions...")
        elif not empty_instruction:
            self.instructions = [DEFAULT_INSTRUCTION]
            logging.warning(f"Using default instructions...")
        else:
            logging.warning(f"Do not use instructions...")
        # instructions should be tailed with a space
        self.instructions = [append_space_if_not_ended_with_space(ins) for ins in self.instructions]
        self.prefix_space = " " if not 'llama' in tokenizer.name_or_path else ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # filtered entry will only contain `src` and `tgt` sequences, though other entries are allowed
        data = self.data[idx]
        src_text = data["src"].rstrip()
        tgt_text = data["tgt"].rstrip()
        
        ins = random.choice(self.instructions) if self.instructions else ""
        preprocessed = tokenize_universal(ins + src_text + self.tokenizer.eos_token, 
                                          self.prefix_space + tgt_text + self.tokenizer.eos_token, self.tokenizer)
        return dict(
            input_ids=preprocessed["input_ids"],
            labels=preprocessed["labels"], 
            # attention_mask=[1] * len(preprocessed["input_ids"])
            # on 05/20: change to ignore EOS/PAD version, see OPT gains any performance?
            attention_mask=[1 if i not in [self.tokenizer.eos_token_id, 
                                       self.tokenizer.pad_token_id] else 0 
                            for i in preprocessed["input_ids"]
                            ]
        )
        
class MixedUpDataset(UniversalJsonlDataset):
    def __init__(self, mixup_ratio=0.5, mixup_index: List[int] = None,
                 *dataset_args, **dataset_kwargs):
        super().__init__(*dataset_args, **dataset_kwargs)
        # pre-init check
        self.mixup_ratio = mixup_ratio
        self.mixup_index = mixup_index
        
        if self.mixup_index:
            # assert len(self.mixup_index) < len(self.data), "Mixup index should be less than dataset size"
            random.shuffle(self.mixup_index)
        
        elif self.mixup_ratio:
            if self.mixup_ratio > 1:
                # expand population set
                self.mixup_index = math.ceil(self.mixup_ratio) * list(range(len(self.data)))
                self.mixup_index = self.mixup_index[:int(self.mixup_ratio * len(self.data))]
                random.shuffle(self.mixup_index)
            elif 0 < self.mixup_ratio <= 1:
                self.mixup_index = random.sample(range(len(self.data)), int(self.mixup_ratio * len(self.data)))
            else:
                raise ValueError("Mixup ratio should be in range (0, 1] or > 1")
            
            
    def __len__(self):
        return len(self.mixup_index)
        
    def __getitem__(self, idx):
        # if mixup_index is provided, we just use it to super sample original dataset
        if self.mixup_index:
            idx = self.mixup_index[idx]
            return super().__getitem__(idx)
        
        
@dataclass
class UniversalDataCollatorForStage1:
    tokenizer: transformers.PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        labels = [instance.pop('labels') for instance in instances]  # 'labels' will prevent tokenizer.pad
        batch = self.tokenizer.pad(
            instances,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # get padded length
        labels = [torch.LongTensor(label) for label in labels]
        if self.pad_to_multiple_of:
            padded_len = batch['input_ids'].shape[1]
            labels.append(torch.empty(padded_len, dtype=torch.long))
        # pad labels with IGNORE_INDEX
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX,
        )
        if self.pad_to_multiple_of:
            # remove dummy tensor
            batch['labels'] = batch['labels'][:-1, :]
        return batch
    


def build_universal_data_modules(tokenizer: transformers.PreTrainedTokenizer, 
                                 data_args,
                                 ):
    assert data_args.train_jsonl_path is not None
    mixup_datasets = []
    train_dataset = UniversalJsonlDataset(data_args.train_jsonl_path, tokenizer=tokenizer, 
                                    instruction_file=data_args.instruction_path,
                                    empty_instruction=data_args.empty_instruction)
    if data_args.dev_jsonl_path is not None:
        eval_dataset = UniversalJsonlDataset(data_args.dev_jsonl_path, tokenizer=tokenizer, 
                                instruction_file=None,
                                empty_instruction=data_args.empty_instruction)
    else:
        eval_dataset = None
        
    if data_args.mix_jsonl_path is not None:
        if data_args.mix_ratio is not None:
            if len(data_args.mix_ratio) == 1:
                data_args.mix_ratio = data_args.mix_ratio * len(data_args.mix_jsonl_path)
            assert len(data_args.mix_jsonl_path) == len(data_args.mix_ratio), f"Length of mix_jsonl_path and mix_ratio should be the same!"
            mix_ratio = data_args.mix_ratio
        else:  # default 0.5
            mix_ratio = [0.5] * len(data_args.mix_jsonl_path)
        for jsonl_file, ratio in zip(data_args.mix_jsonl_path, mix_ratio):
            mixup_datasets.append(MixedUpDataset(jsonl_file=jsonl_file, mixup_ratio=ratio,
                                                tokenizer=tokenizer, 
                                                    instruction_file=None,
                                                    empty_instruction=data_args.empty_instruction))
        
    data_collator = UniversalDataCollatorForStage1(tokenizer=tokenizer,
                                                   padding=True,
                                                   max_length=tokenizer.model_max_length,
                                                   pad_to_multiple_of=data_args.pad_to_multiple_of,
                                                   )
    if mixup_datasets:
        print(f"Using {len(mixup_datasets)} mixup datasets!")
        train_dataset = ConcatDataset([train_dataset] + mixup_datasets)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


if __name__ == '__main__':
    # run: PYTHONPATH=src/ python src/common_utils/universal_dataset_utils.py
    from torch.utils.data import DataLoader
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/v-zilinxiao/code/transformers/llama_7B_EL_full_evalsample_0.5")
    dataset = UniversalJsonlDataset("/home/v-zilinxiao/data/dataset/benchmark_test/debug.jsonl", 
                                    tokenizer)
    # dataset = MixedUpDataset(jsonl_file='/home/v-zilinxiao/data/dataset/benchmark_test/aida_train.jsonl', 
    #                          tokenizer=tokenizer, 
    #                          mixup_ratio=0.5, mixup_index=None,)
    data_collator = UniversalDataCollatorForStage1(tokenizer=tokenizer, pad_to_multiple_of=None)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator, num_workers=0)
    
    # manual seeing passed; wait for unit test
    for i, batch in enumerate(dataloader):
        print(batch)
        print("=" * 20 + f" {i} " + "=" * 20)
        if i > 10:
            break
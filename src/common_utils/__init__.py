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


# not used
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]  # problematic as in llama, eos_token is also pad_token
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_single(source: str, target: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # source = preprocess_input(source)  # added 4-8: DO NOT preprocess target, collected dataset already preprocessed

    example = source + target  # concat input-output -> example
    example_tokenized = _tokenize_fn([example], tokenizer)  # ins + input + output
    source_tokenized = _tokenize_fn([source], tokenizer)  # source was only used to compute ignore_len

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)  # copy input_ids to labels, constructing a CLM task
    source_len = source_tokenized["input_ids_lens"][0]

    labels[:source_len] = IGNORE_INDEX  # prompt part gets ignored
    return dict(input_ids=input_ids, labels=labels)  # for this sample only


def append_space_if_not_ended_with_space(string: str) -> str:
    """Append a space if the string does not end with a space."""
    if string.endswith(" "):
        return string
    else:
        return string + " "


class JsonlDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer: transformers.PreTrainedTokenizer = None, 
                 instruction_file=None, empty_instruction=False) -> None:
        self.tokenizer = tokenizer
        with open(jsonl_file, "r") as f:
            self.data: List[dict] = [json.loads(line) for line in f]  
            # TODO: use Serilized Toolbox to save memory?
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # filtered entry will only contain `src` and `tgt` sequences, though other entries are allowed
        data = self.data[idx]
        src_text = data["src"].rstrip()
        tgt_text = data["tgt"].rstrip()
        
        ins = random.choice(self.instructions) if self.instructions else ""
        preprocessed = preprocess_single(ins + src_text + self.tokenizer.eos_token,
                                         tgt_text + self.tokenizer.eos_token, self.tokenizer)
        return dict(
            input_ids=preprocessed["input_ids"],
            labels=preprocessed["labels"], 
        )


class MixedUpDataset(JsonlDataset):
    def __init__(self, mixup_ratio=0.5, mixup_index: List[int] = None,
                 *dataset_args, **dataset_kwargs):
        super().__init__(*dataset_args, **dataset_kwargs)
        # pre-init check
        self.mixup_ratio = mixup_ratio
        self.mixup_index = mixup_index
        
        if self.mixup_index:
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
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  
            # do not attend to padded tokens; however, pad_token_id is the same with eos_token_id in most settings
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    mixup_datasets = []
    if data_args.train_jsonl_path is not None:
        print(f"Using jsonl dataset!!!")
        train_dataset = JsonlDataset(data_args.train_jsonl_path, tokenizer=tokenizer, 
                                        instruction_file=data_args.instruction_path,
                                       empty_instruction=data_args.empty_instruction)
        if data_args.dev_jsonl_path is not None:
            eval_dataset = JsonlDataset(data_args.dev_jsonl_path, tokenizer=tokenizer, 
                                    instruction_file=None,
                                    empty_instruction=data_args.empty_instruction)
        else:
            eval_dataset = None
        
        # mix_jsonl_path: Optional[List[str]] = field(default=None)
        # mix_ratio: Optional[List[float]] = field(default=None)
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
            
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if mixup_datasets:
        print(f"Using {len(mixup_datasets)} mixup datasets!")
        train_dataset = ConcatDataset([train_dataset] + mixup_datasets)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


if __name__ == '__main__':
    # load a train_dataset and compare the input_ids and labels
    # run: PYTHONPATH=src/ python src/common_utils/__init__.py
    from torch.utils.data import DataLoader
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/v-zilinxiao/code/transformers/llama_7B_EL_full_evalsample_0.5")
    dataset = JsonlDataset("/home/v-zilinxiao/data/dataset/benchmark_test/debug.jsonl", 
                            tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator, num_workers=0)

    for i, batch in enumerate(dataloader):
        print(batch)
        print("=" * 20 + f" {i} " + "=" * 20)
        if i > 10:
            break
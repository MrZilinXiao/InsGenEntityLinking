from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Trainer
import transformers

if not transformers.__version__.endswith("dev0"):
    raise ImportError("You are not using the latest version of transformers.")

from common_utils.universal_dataset_utils import (
    build_universal_data_modules,
)
from datetime import datetime

from peft import (
    LoraConfig,
    get_peft_model,
)
import time


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    empty_instruction: bool = field(default=False)
    train_jsonl_path: Optional[str] = field(default=None)
    dev_jsonl_path: Optional[str] = field(default=None)
    # update on 05/17, add mixedup dataset support
    mix_jsonl_path: Optional[List[str]] = field(default=None)
    mix_ratio: Optional[List[float]] = field(default=None)
    # depreciated below
    train_source_path: Optional[str] = field(default=None)
    train_target_path: Optional[str] = field(default=None)
    dev_source_path: Optional[str] = field(default=None)
    dev_target_path: Optional[str] = field(default=None)
    # depreciated above
    instruction_path: Optional[str] = field(default=None)
    pad_to_multiple_of: Optional[int] = field(default=None)
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # additional lora args
    lora_enabled: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

    def __post_init__(self):
        super().__post_init__()
        if self.report_to == "wandb":
            # prepend mmddyyyy format datetime before the run_name
            self.run_name = datetime.now().strftime("%m%d%Y") + "_" + self.run_name


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # if tokenizer.pad_token is None:  # only happens in llama_hf_7B
    tokenizer.pad_token = tokenizer.eos_token  # override all pad_token to eos_token

    if training_args.lora_enabled:
        assert 'llama' in model_args.model_name_or_path, "peft lora only tested for llama models"
        # prepare lora model
        # model = prepare_model_for_int8_training(model)  # for int8 training; not in our case
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=['q_proj', 'v_proj'],
            lora_dropout=training_args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()
        time.sleep(1)

    data_module = build_universal_data_modules(tokenizer=tokenizer, data_args=data_args)
    # data_module: dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # how to load lora model?
    # official way:
    # 1. model.save_pretrained("output_dir")
    # 2. config = PeftConfig.from_pretrained("output_dir")
    # 3. model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    # 4. model = PeftModel.from_pretrained(model, "output_dir")
    
    trainer.save_model(training_args.output_dir)  # final optimizer is not saved (commented on 05/12)


if __name__ == "__main__":
    train()

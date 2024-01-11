# InsGenEntityLinking
Official Implementation of the EMNLP 2023 paper on "Instructed Language Models with Retriever Are Powerful Entity Linkers". Code, checkpoints and documents are being organized and kept being updated.

## Environment
During the time of development, the implementation of LLaMA in `transformers` is not stable, so we provided an adapted version in this repo. Be sure to uninstall the local `transformers` package before running the code, running it in a clean environment or changing the order of `PYTHONPATH` so that local packages have higher priority than the installed ones.

## Dataset Preparation

### Wikipedia Training Data

1. Download the latest Wikipedia English dump from [here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2).

2. Install wikipedia2vec from PyPI.:
    ```bash
    pip install wikipedia2vec
    ```

3. Run the following command to use `wikipedia2vec` to build dump db from the downloaded Wikipedia dump. Dump db allows parallel access to the Wikipedia dump.
    ```bash
    wikipedia2vec build-dump-db enwiki-latest-pages-articles.xml.bz2 enwiki-latest-pages-articles.db
    ```

4. Run the following command to convert from dump db to jsonl input-output format of InsGenEL. You may need to change the hard-coded paths in the script.
    ```bash
    python data_scripts/wiki_dump.py
    ```

Now you have a jsonl file containing the training pairs for InsGenEL.

### Entity Linking Evaluation Data

We follow the procedures of [elevant](https://github.com/ad-freiburg/elevant) to prepare the evaluation data. To begin, clone the repo first:

```bash
git clone https://github.com/ad-freiburg/elevant
```

Evaluation data should be in it. Run the following command to download the evaluation mappings:

```bash
make download_all
```

## Training

```bash
deepspeed --num-gpus=8 universal_train.py --model_name_or_path ./llama_hf_7B --train_jsonl_path /home/v-zilinxiao/data/dataset/wiki_full_eval_only/wiki_eval_sample_0.2.jsonl --empty_instruction False --bf16 False --output_dir ./llama_7B_final_checkpoint/ --num_train_epochs 1 --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 1 --save_strategy steps --save_steps 50000 --save_total_limit 10 --learning_rate 2e-5 --weight_decay 0. --lr_scheduler_type polynomial --warmup_ratio 0.03 --logging_steps 10 --deepspeed ./deepspeed_configs/ds_config_zero3.json --fp16 --report_to wandb --run_name llama_7B_final_checkpoint --tf32 False
```

Above command should take 61 hours on 8 V100-SXM-32GB GPUs with NVLink.

## Evaluation
As we opt in the modern toolkit `elevant` for evaluating entity linking performance, we build an persistent API for `elevant` so that it receives the output of InsGenEL and returns the evaluation results.

First, open a terminal and run the following command to start the API server:

```bash
PYTHONPATH=. python eval/persistent_el_eval.py
```

To faciliate the evaluation, we provide a watch-dog script that consistently monitors the output directory of InsGenEL and evaluate the latest model on the evaluation data. To use it, open another terminal and run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python eval/universal_offline_eval.py --watch_path ./llama_7B_EL_full_evalsample_0.5 --checkpoint_name .
```

Result log files will be saved in the format of "elevant_results_*.log" in the checkpoint directory.

## Acknowledgement
We thank the authors of [elevant](https://github.com/ad-freiburg/elevant), [wikipedia2vec](https://github.com/wikipedia2vec/wikipedia2vec) and [GENRE](https://github.com/facebookresearch/GENRE) and parts of our code are borrowed from them with minor modifications.

## Citation

```bibtex
@inproceedings{xiao-etal-2023-instructed,
    title = "Instructed Language Models with Retrievers Are Powerful Entity Linkers",
    author = "Xiao, Zilin  and
      Gong, Ming  and
      Wu, Jie  and
      Zhang, Xingyao  and
      Shou, Linjun  and
      Jiang, Daxin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.139",
    doi = "10.18653/v1/2023.emnlp-main.139",
    pages = "2267--2282"
}
```

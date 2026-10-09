# How Tokenization Affects Small Multilingual Language Models

This project compares three ways of tokenizing text: a character tokenizer and two byte-level BPE tokenizers (vocabulary sizes 2,000 and 10,000). For each tokenizer I trained a small Transformer language model from scratch on English, Turkish and Chinese Wikipedia text, and I compared the models with bits per character (BPC). The report explains the experiment and the results.

## Repository structure

```
src/        all the Python scripts
models/     trained tokenizers (.json) and trained models (.pt)
results/    CSV files with the results
plots/      figures used in the report
```

The data is not in this repository. The scripts read it from `/srv/data/lt2326-h26/a1/` on mltgpu (`train/`, `valid/`, `test/` and `tokenizer/balanced.txt`).

## Requirements

Python 3, PyTorch, `tokenizers` (Hugging Face), pandas and matplotlib. Training was done on a GPU on mltgpu-2.

## Paths

The scripts have the paths to my own folders on mltgpu (`/home/gusgoutdi@GU.GU.SE/a1_tokenization/...`). To run them somewhere else, change these paths at the top of each script.

## How to reproduce the experiments

Run the scripts from the `src/` folder, in this order.

**1. Train the tokenizers**

- `char_tokenizer.py` builds the character vocabulary from `balanced.txt` (9,443 tokens, including `<UNK>`). It is imported by the other scripts and does not need to be run.
- `bpe_tokenizer.py` trains a byte-level BPE tokenizer on `balanced.txt`. The vocabulary size and the output file name are set in the script. I ran it once with 2,000 (`bpe_2k.json`) and once with 10,000 (`bpe_10k.json`).

**2. Tokenizer statistics and examples**

- `tokenizer_stats.py` counts tokens and characters per token on the validation files (`results/tokenizer_stats.csv`).
- `tokenization_examples.py` prints how the three tokenizers split the first sentences of each validation file.

**3. Train the models**

- `train.py` trains one model. Set `tokenizer_name` in the script to `char`, `bpe_2k` or `bpe_10k` and run it three times. All models use the same settings: 2 layers, hidden size 256, 4 heads, feed-forward size 1024, dropout 0.1, context length 128, AdamW with learning rate 3e-4, batch size 32, 3,000 updates and seed 2326. The model is defined in `model.py` and the data in `dataset.py`. The script saves `models/<name>_final.pt` and the loss curve in `plots/`.

**4. Evaluate the saved models**

- `evaluate.py` computes the test BPC. Set `model_name` and `language` in the script (`en`, `tr` or `zh`). The BPC of every model and language is collected in `results/bpc_results.csv`. The overall column is the average of the three language BPC values.

**5. Analyses**

- `parameter_counts.py` counts the parameters of each model (`results/parameter_counts.csv`).
- `vocabulary_allocation.py` shows how the vocabulary of each BPE tokenizer is divided between the languages, with a threshold of 80% (`results/vocabulary_allocation.csv` and `results/vocabulary_examples.csv`).
- `chinese_fragmentation.py` checks how many Chinese characters are split into more than one token (`results/chinese_fragmentation.csv`).
- `error_analysis.py` compares the three models on the first three sentences of the Chinese test file (`results/error_analysis.csv`).

**6. Figures**

- `plot_results.py` makes the figures for the characters per token, the BPC and the vocabulary allocation. The training curves are made by `train.py`.

## Where each table and figure comes from

| In the report | Script | Output |
|---|---|---|
| Table 1, Figure 1 | `tokenizer_stats.py`, `plot_results.py` | `results/tokenizer_stats.csv` |
| Table 2 | `tokenization_examples.py` | printed output |
| Table 3 | `parameter_counts.py` | `results/parameter_counts.csv` |
| Figure 2 | `train.py` | `plots/final_<name>_training_curve.png` |
| Table 4, Figure 3 | `evaluate.py`, `plot_results.py` | `results/bpc_results.csv` |
| Table 5 | `vocabulary_allocation.py` | `results/vocabulary_allocation.csv` |
| Chinese fragmentation (Section 5) | `chinese_fragmentation.py` | `results/chinese_fragmentation.csv` |
| Table 6, Figure 4 | `error_analysis.py` | `results/error_analysis.csv` |

## Trained models

The trained models are in `models/` (`char_final.pt`, `bpe_2k_final.pt`, `bpe_10k_final.pt`), together with the two BPE tokenizers (`bpe_2k.json`, `bpe_10k.json`). Each model is about 10 to 27 MB.

# Tokenizer Triner

## Overview

The `TokenizerProcessor` is a Python utility designed for processing text datasets and training new tokenizers. It allows for the creation of tokenizers with custom vocabulary sizes, visualization of token distributions, and saving the trained tokenizer for reuse. This tool is built with Hugging Face's `transformers` and `datasets` libraries, making it suitable for modern NLP pipelines.

---

## Features

1. **Dataset Loading**: Load datasets from the Hugging Face `datasets` library.
2. **Tokenizer Training**: Train a new tokenizer using an existing pretrained tokenizer as a base.
3. **Save Tokenizer**: Save the newly trained tokenizer for later use.
4. **Token Distribution Visualization**: Visualize token length distributions across dataset splits.

---

## Installation

To use this script, ensure the following Python libraries are installed:

- `pandas`
- `datasets`
- `transformers`
- `matplotlib`
- `seaborn`
- `argparse`

Install the libraries using `pip`:

```bash
pip install pandas datasets transformers matplotlib seaborn
```

---

## Usage

### Command-Line Interface

Run the script using the command line with the following arguments:

```bash
python tokenizer_processor.py --dataset_name <DATASET_NAME> \
                              --old_tokenizer_name <PRETRAINED_TOKENIZER> \
                              --text_field <TEXT_FIELD_NAME> \
                              --vocab_size <VOCAB_SIZE> \
                              --train_split <DATASET_SPLIT> \
                              --save_path <SAVE_PATH> \
                              [--plot]
```

#### Arguments:
- `--dataset_name`: Name of the dataset from Hugging Face (e.g., `ag_news`).
- `--old_tokenizer_name`: Name of the pretrained tokenizer (e.g., `bert-base-uncased`).
- `--text_field`: Name of the text field in the dataset (e.g., `text` or `combined_seq`).
- `--vocab_size`: Vocabulary size for the new tokenizer (e.g., `32768`).
- `--train_split`: Dataset split for training (e.g., `train`, `test`, `validation`).
- `--save_path`: Path to save the trained tokenizer.
- `--plot`: (Optional) Include this flag to visualize token length distributions.

### Example:

```bash
python tokenizer_processor.py --dataset_name ag_news \
                              --old_tokenizer_name bert-base-uncased \
                              --text_field text \
                              --vocab_size 32000 \
                              --train_split train \
                              --save_path ./new_tokenizer \
                              --plot
```

---

## Programmatic Usage

The `TokenizerProcessor` class can be used directly in Python scripts for more control. 

```python
from tokenizer_processor import TokenizerProcessor

processor = TokenizerProcessor(
    dataset_name="ag_news",
    old_tokenizer_name="bert-base-uncased",
    text_field="text",
    vocab_size=32000,
    train_split="train"
)

processor.load_dataset()
processor.train_new_tokenizer()
processor.save_tokenizer("./new_tokenizer")

# Optional: Plot token distribution
TokenizerProcessor.plot_token_distribution(
    tokenizer_name="bert-base-uncased",
    dataset_name="ag_news",
    text_field="text"
)
```

---

## Functionality

### 1. `load_dataset()`
Loads the dataset using the Hugging Face `datasets` library.

### 2. `train_new_tokenizer()`
Trains a new tokenizer with a specified vocabulary size using the given text field.

### 3. `save_tokenizer(save_path)`
Saves the trained tokenizer to a specified directory.

### 4. `plot_token_distribution(tokenizer_name, dataset_name, text_field)`
Generates a histogram of token length distributions for the specified dataset splits.

---

## Dependencies

- Python 3.7+
- Hugging Face `transformers` and `datasets`
- Visualization libraries (`matplotlib`, `seaborn`)

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.

---

## Contribution

Contributions are welcome! Feel free to submit issues or pull requests for improvements or new features.

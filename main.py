import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

class TokenizerProcessor:
    def __init__(self, dataset_name, old_tokenizer_name, text_field="combined_seq", vocab_size=32768, train_split="train"):
        """
        Initialize the TokenizerProcessor with dataset name, tokenizer name, text field, and vocabulary size.

        Args:
            dataset_name (str): The name of the dataset to process.
            old_tokenizer_name (str): The name of the pretrained tokenizer.
            text_field (str): The name of the text field in the dataset to tokenize. Defaults to 'combined_seq'.
            vocab_size (int): The size of the vocabulary for the new tokenizer. Defaults to 32768.
            train_split (str): The split of the dataset to use for training. Defaults to 'train'.
        """
        self.dataset_name = dataset_name
        self.old_tokenizer_name = old_tokenizer_name
        self.text_field = text_field
        self.vocab_size = vocab_size
        self.train_split = train_split
        self.raw_datasets = None
        self.tokenizer = None

    def load_dataset(self):
        """Load the dataset using the specified dataset name."""
        self.raw_datasets = load_dataset(self.dataset_name)

    def train_new_tokenizer(self):
        """
        Train a new tokenizer using the old tokenizer and the training corpus.
        """
        if not self.raw_datasets:
            raise ValueError("Dataset not loaded. Please call load_dataset first.")

        if self.train_split not in self.raw_datasets:
            raise ValueError(f"Split '{self.train_split}' not found in the dataset.")

        old_tokenizer = AutoTokenizer.from_pretrained(self.old_tokenizer_name)
        training_corpus = (
            self.raw_datasets[self.train_split][i : i + 1000][self.text_field]
            for i in range(0, len(self.raw_datasets[self.train_split]), 1000)
        )
        self.tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, self.vocab_size)

    def save_tokenizer(self, save_path):
        """
        Save the newly trained tokenizer to a specified path.

        Args:
            save_path (str): Path to save the tokenizer.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained. Please call train_new_tokenizer first.")

        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def plot_token_distribution(tokenizer_name, dataset_name, text_field="combined_seq"):
        """
        Plot the token distribution for different splits of a Hugging Face dataset.

        Args:
            tokenizer_name (str): Name of the tokenizer from Hugging Face.
            dataset_name (str): Name of the dataset from Hugging Face.
            text_field (str): The name of the text field in the dataset to tokenize. Defaults to 'combined_seq'.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dataset = load_dataset(dataset_name)

        for split, split_data in dataset.items():
            print(f"Processing split: {split}")
            token_lengths = split_data.map(
                lambda example: {"token_count": len(tokenizer(example[text_field], truncation=True, padding=False).input_ids)},
                batched=False
            )

            token_counts = token_lengths["token_count"]
            sns.histplot(token_counts, kde=True, bins=30)
            plt.title(f"Token Length Distribution for Split: {split}")
            plt.xlabel("Number of Tokens")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Processor")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on HF to process.")
    parser.add_argument("--old_tokenizer_name", type=str, required=True, help="Name of the pretrained tokenizer.")
    parser.add_argument("--text_field", type=str,equired=True, help="Name of the text field in the dataset.")
    parser.add_argument("--vocab_size", type=int, equired=True, help="Vocabulary size for the new tokenizer.")
    parser.add_argument("--train_split", type=str, default="train", help="Dataset split to use for training (e.g., train, test, validation).")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the new tokenizer.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot token distributions.")

    args = parser.parse_args()

    processor = TokenizerProcessor(
        dataset_name=args.dataset_name,
        old_tokenizer_name=args.old_tokenizer_name,
        text_field=args.text_field,
        vocab_size=args.vocab_size,
        train_split=args.train_split
    )

    processor.load_dataset()
    processor.train_new_tokenizer()
    processor.save_tokenizer(args.save_path)

    if args.plot:
        TokenizerProcessor.plot_token_distribution(
            tokenizer_name=args.old_tokenizer_name,
            dataset_name=args.dataset_name,
            text_field=args.text_field
        )

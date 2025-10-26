import yaml
from datasets import load_dataset, concatenate_datasets

class SnliMnliPreprocessor:
    """
    Preprocesses SNLI and MNLI datasets for Multiple Negative Ranking.

    Concatenates SNLI and MNLI datasets, 
    removes uncertain or invalid labels (where label = -1), 
    and keeps only positive (entailment) pairs.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.dataset_train = None
        self.dataset_val = None

        # Load datasets
        self.raw_datasets = {
            "nli": {
                "train": load_dataset(self.cfg["datasets"]["all-nli"]["name"], "pair-class", split="train"),
                "val": load_dataset(self.cfg["datasets"]["all-nli"]["name"], "pair-class", split="dev"),
            },
            "mnli": {
                "train": load_dataset(self.cfg["datasets"]["mnli"]["name"], split="train"),
                "val": load_dataset(self.cfg["datasets"]["mnli"]["name"], split="validation"),
            },
        }

    def _filter_mnli_columns(self):
        """Remove unnecessary columns from MNLI."""
        for split in ["train", "val"]:
            if "idx" in self.raw_datasets["mnli"][split].column_names:
                self.raw_datasets["mnli"][split] = self.raw_datasets["mnli"][split].remove_columns(["idx"])

    def _concat_datasets(self):
        """Ensure same data types and concatenates datasets."""
        for split in ["train", "val"]:
            nli = self.raw_datasets["nli"][split]
            mnli = self.raw_datasets["mnli"][split]

            nli = nli.cast(mnli.features)  # align feature schemas
            combined = concatenate_datasets([nli, mnli])

            if split == "train":
                self.dataset_train = combined
            else:
                self.dataset_val = combined

    def _filter_positive_pairs(self, dataset):
        """
        Removes uncertain (-1) samples and keeps only positive (label == 0) pairs.
        """
        dataset = dataset.filter(lambda x: x["label"] != -1)
        return dataset.filter(lambda x: x["label"] == 0)

    def preprocess(self):
        """
        Returns:
            train_set (type dataset): filtered positive pairs for training
            val_set (type dataset): filtered positive pairs for validation
        """
        self._filter_mnli_columns()
        self._concat_datasets()

        train_set = self._filter_positive_pairs(self.dataset_train)
        val_set = self._filter_positive_pairs(self.dataset_val)
        return train_set, val_set

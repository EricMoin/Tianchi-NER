import torch
from torch.utils.data import Dataset
from conll_reader import ConllEntity
from transformers import AutoTokenizer


class NERDataEntity:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class NERDataset(Dataset):
    def __init__(self, data: list[ConllEntity], tokenizer: AutoTokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        unique_labels = set()
        for example in data:
            unique_labels.update(example.labels)
        self.label_map = {label: i for i, label in enumerate(
            sorted(unique_labels))}

    def __getitem__(self, index: int) -> dict:
        # Initialize empty lists
        tokens = []
        labels = []
        is_prompt = False

        # Process all tokens/labels
        for token, label in zip(self.data[index].tokens, self.data[index].labels):
            if token == "<EOS>":
                is_prompt = True
                continue
            if not is_prompt:
                tokens.append(token)
                labels.append(label)

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=150,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Convert labels to tensor with fallback to "O" for unknown labels
        label_ids = []
        for label in labels:
            if label in self.label_map:
                label_ids.append(self.label_map[label])
            else:
                # Use "O" (Outside) tag for unknown labels
                label_ids.append(self.label_map["O"])
                print(
                    f"Warning: Unknown label '{label}' found, using 'O' instead")

        # Pad label_ids to match input_ids length
        padded_labels = label_ids + \
            [self.label_map["O"]] * (150 - len(label_ids))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(padded_labels)
        }

    def __len__(self):
        return len(self.data)

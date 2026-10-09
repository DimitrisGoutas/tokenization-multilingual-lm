import torch
from torch.utils.data import Dataset

class LanguageModelDataset(Dataset):
    def __init__(self, encode, data_paths, context_length=128):
        self.context_length = context_length
        # read the data
        sentences = []
        for data_path in data_paths:
            with open(data_path, "r", encoding="utf-8") as text:
                # takes all the lines and adds them to a list
                sentences.extend(text.read().splitlines())
        # tokenize all sentences
        self.token_ids = []
        for sentence in sentences:
            token_ids = encode(sentence)
            self.token_ids.extend(token_ids)
        self.token_ids = torch.tensor(self.token_ids, dtype=torch.long)

    def __len__(self):
        # each example needs 128 input tokens + 1 target token
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):
        # take 129 tokens
        chunk = self.token_ids[index:index + self.context_length + 1]
        # first 128 tokens
        input_tokens = chunk[:-1]
        target_tokens = chunk[1:]

        return input_tokens, target_tokens

# I used non-overlapping chunks here to avoid evaluating the same tokens multiple times
class EvaluationDataset(LanguageModelDataset):

    def __len__(self):
        # chunks start every context_length tokens, with no overlap,
        # and each chunk needs one extra token for the last target
        return (len(self.token_ids) - 1) // self.context_length

    def __getitem__(self, index):
        start = index * self.context_length
        chunk = self.token_ids[start:start + self.context_length + 1]
        input_tokens = chunk[:-1]
        target_tokens = chunk[1:]

        return input_tokens, target_tokens
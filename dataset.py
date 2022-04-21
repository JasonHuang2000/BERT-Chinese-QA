import os
import json
from itertools import chain
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class MultiChoiceDataset(Dataset):
    
    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: AutoTokenizer,
        max_len: int = 512,
        pad_to_max_len: bool = False,
    ):
        assert split in ['train', 'valid', 'test'], \
            "property `split` should be either \'train\', \'valid\', or \'test\'"

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len

        self.context_path = os.path.join(data_root, 'context.json')
        self.data_path = os.path.join(data_root, f'{split}.json')
        self.data = self._preprocessed()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    def _preprocessed(self) -> List[Dict]:
        with open(self.context_path, 'r', encoding='utf-8') as context_f:
            raw_context: List = json.load(context_f)
        with open(self.data_path, 'r', encoding='utf-8') as data_f:
            raw_data: List[Dict] = json.load(data_f)

        ret = [{
            'question': [sample['question']] * len(sample['paragraphs']),
            'context': [raw_context[idx] for idx in sample['paragraphs']],
            'label': sample['paragraphs'].index(sample['relevant']),
        } for sample in raw_data]

        return ret

    def collate(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(samples)
        num_choices = len(samples[0]['context'])
        
        # extract data
        question_set: List[List] = [sample['question'] for sample in samples]
        context_set: List[List] = [sample['context'] for sample in samples]
        labels: List = [sample['label'] for sample in samples]

        # flatten input
        questions = list(chain(*question_set))
        contexts = list(chain(*context_set))

        # tokenize
        batch = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length" if self.pad_to_max_len else "longest",
            return_tensors="pt"
        )

        # un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # add lable
        batch['labels'] = torch.tensor(labels)

        return batch
import os
import json
import numpy as np
from itertools import chain
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import datasets

class MultiChoiceDataset(Dataset):
    
    def __init__(
        self,
        data_files: Dict[str, str],
        split: str,
        tokenizer: AutoTokenizer,
        max_len: int = 512,
        pad_to_max_len: bool = False,
    ):
        assert split in ['train', 'valid', 'test'], \
            "property `split` should be either \'train\', \'valid\', or \'test\'"
        assert 'context' in data_files, "please provide context file to build the dataset"
        assert split in data_files, "please provide data file of corresponding split to build the dataset"

        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len

        self.context_path = data_files['context']
        self.data_path = data_files[split]
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

        if self.split == 'test':
            ret = [{
                'question': [sample['question']] * len(sample['paragraphs']),
                'context': [raw_context[idx] for idx in sample['paragraphs']],
            } for sample in raw_data]
        else:
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
        if self.split != 'test':
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
        if self.split != 'test':
            batch['labels'] = torch.tensor(labels)

        return batch

def preprocces_data_files(
    data_files: Dict[str, str], 
    splits: List[str],
    context_preds: np.array = None,
) -> Dict[str, str]:

    assert 'context' in data_files, "please provide context file to build the dataset"

    context_file_path = data_files['context']
    with open(context_file_path, 'r', encoding='utf-8') as context_f:
        raw_context = json.load(context_f)

    new_data_files = {}

    for split in splits:

        assert split in ['train', 'valid', 'test', 'test_all_context'], \
            "`splits` should only contain \'train\', \'valid\', \'test\' or \'test_all_context\'"

        source_file_path = data_files[split]
        data_root = os.path.dirname(source_file_path)
        target_file_path = os.path.join(data_root, f'{split}_qa.json')
        new_data_files[split] = target_file_path
        if os.path.exists(target_file_path) and 'test' not in split:
            print(f'{target_file_path} already exists, skipping...')
            continue
        with open(source_file_path, 'r', encoding='utf-8') as source_f, \
            open(target_file_path, 'w', encoding='utf-8') as target_f:
            data = json.load(source_f)
            if split == 'test':
                for idx, sample in enumerate(data):
                    sample['context'] = raw_context[sample['paragraphs'][context_preds[idx]]]
                    sample.pop('paragraphs')
            elif split == 'test_all_context':
                for idx, sample in enumerate(data):
                    sample['context'] = ''.join([raw_context[idx] for idx in sample['paragraphs']])
                    sample.pop('paragraphs')
            else:
                for sample in data:
                    sample['context'] = raw_context[int(sample['relevant'])] 
                    sample.pop('relevant')
                    sample.pop('paragraphs')
            json.dump({
                'data': data,
            }, target_f, ensure_ascii=False, indent=4)
            print(f'processed {split} data file')

    return new_data_files

def load_qa_dataset(
    data_files: Dict[str, str],
) -> datasets.Dataset:
    return datasets.load_dataset(
        'json', 
        data_files=data_files,
        field='data',
    )
import sys
import os
import numpy as np

from transformers import (
    AutoModelForMultipleChoice, 
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments, 
    Trainer,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments
from dataset import MultiChoiceDataset

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def main():
    
    # utilize huggingface argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # fix random seed
    set_seed(training_args.seed)

    # pretrained model and tokenizer
    model = AutoModelForMultipleChoice.from_pretrained(model_args.pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.pretrained_name
    )

    # datasets
    train_dataset = MultiChoiceDataset(
        data_files={
            'context': data_args.context_file,
            'train': data_args.train_file,
        },
        split='train',
        tokenizer=tokenizer,
    )
    eval_dataset = MultiChoiceDataset(
        data_files={
            'context': data_args.context_file,
            'valid': data_args.valid_file,
        },
        split='valid',
        tokenizer=tokenizer
    )

    # initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.collate,
        compute_metrics=compute_metrics,
    )

    # training
    trainer.train(
        resume_from_checkpoint=model_args.ckpt_path)


if __name__ == '__main__':
    main()
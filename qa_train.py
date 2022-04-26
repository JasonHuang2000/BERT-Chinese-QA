import sys
import os

from transformers import (
    AutoModelForQuestionAnswering,
    BertConfig,
    AutoTokenizer,
    HfArgumentParser,
    EvalPrediction,
    DefaultDataCollator,
    TrainingArguments, 
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments
from dataset import preprocces_data_files, load_qa_dataset
from trainer import QuestionAnsweringTrainer
from utils import postprocess_qa_predictions

def main():
    
    # utilize huggingface argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # fix random seed
    set_seed(training_args.seed)

    # model and tokenizer
    if model_args.from_scratch:
        config = BertConfig(**model_args.bert_config)
        model = AutoModelForQuestionAnswering.from_config(config)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.ckpt_path if model_args.ckpt_path else model_args.pretrained_name
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.pretrained_name
    )

    # datasets
    new_data_files = preprocces_data_files(
        data_files={
            'context': data_args.context_file,
            'train': data_args.train_file,
            'valid': data_args.valid_file,
        },
        splits=['train', 'valid'],
    )
    qa_dataset = load_qa_dataset(
        data_files=new_data_files,
    )

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples['question' if pad_on_right else 'context'],
            examples['context' if pad_on_right else 'question'],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_len,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_len else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

         # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answer"][sample_index]
            
            # Start/end character index of the answer in the text.
            start_char = answers["start"]
            end_char = start_char + len(answers["text"])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_valid_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['question' if pad_on_right else 'context'],
            examples['context' if pad_on_right else 'question'],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_len,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_len else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    column_names = qa_dataset['train'].column_names
    train_dataset = qa_dataset['train'].map(
        prepare_train_features,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )
    valid_dataset = qa_dataset['valid'].map(
        prepare_valid_features,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answer"]["text"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(eval_predictions):
        pred, label = eval_predictions
        cnt = 0
        for p, l in zip(pred, label):
            if p['prediction_text'] == l['answers']:
                cnt += 1
        return {"accuracy": cnt / len(label)}

    # initialize trainer
    data_collator = DefaultDataCollator() 
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        eval_examples=qa_dataset['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # training
    trainer.train(
        resume_from_checkpoint=model_args.ckpt_path)


if __name__ == '__main__':
    main()
import sys
import os
import csv
import torch

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    DefaultDataCollator,
    EvalPrediction,
)

from arguments import InferenceModelArguments, DataTrainingArguments
from dataset import preprocces_data_files, load_qa_dataset
from trainer import QuestionAnsweringTrainer
from utils import postprocess_qa_predictions

def main():
    
    # utilize huggingface argument parser
    parser = HfArgumentParser((InferenceModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    # pretrained model and tokenizer
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.qa_ckpt_path if model_args.qa_ckpt_path else model_args.qa_pretrained_name
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        model_args.qa_tokenizer_name if model_args.qa_tokenizer_name else model_args.qa_pretrained_name
    )

    # fix random seed
    torch.manual_seed(777)

    new_data_files = preprocces_data_files(
        data_files={
            'context': data_args.context_file,
            'test_all_context': data_args.test_file,
        },
        splits=['test_all_context'],
    )
    qa_dataset = load_qa_dataset(
        data_files=new_data_files,
    )

    def prepare_test_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = qa_tokenizer.padding_side == "right"
        tokenized_examples = qa_tokenizer(
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

    column_names = qa_dataset['test_all_context'].column_names
    qa_test_dataset = qa_dataset['test_all_context'].map(
        prepare_test_features,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )

    data_collator = DefaultDataCollator() 

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            prefix=stage,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        return EvalPrediction(predictions=formatted_predictions, label_ids=None)

    qa_trainer = QuestionAnsweringTrainer(
        model=qa_model,
        data_collator=data_collator,
        post_process_function=post_processing_function,
    )
    print("Begin question answering...")
    result = qa_trainer.predict(
        predict_dataset=qa_test_dataset,
        predict_examples=qa_dataset['test_all_context']
    )
    
    with open(data_args.output_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        rows = []
        for pred in result.predictions:
            rows.append([pred["id"], pred["prediction_text"]])
        writer.writerows(rows)

if __name__ == '__main__':
    main()
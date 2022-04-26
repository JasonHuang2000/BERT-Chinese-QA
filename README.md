# BERT Chinese QA

## Prerequisites
- Python >= 3.8
- `pip install -r requirements.txt` to install other dependencies

## Data

A context file `context.json` should be provided, containing all the relevant context with following structure,

```JSON
[
    "context1",
    "context2",
    ...
]
```

Two files `train.json` and `valid.json` should be provided for training. The `JSON` file should contain a list of data sample with structure as follow,

```JSON
[
    {
        "id": "9f0860355c55e113d615cc88b3fa9420",
        "question": "伊利諾大學是哪一個國家的大學?",
        "paragraphs": [
            6037,
            4555,
            5641,
            1029
        ],
        "relevant": 4555,
        "answer": {
        "text": "美國",
        "start": 15
        }
    },
    ...
]
```

For inference, a file `test.json` should be provided, and will only contains question and relevant context index.

```JSON
[
    {
        "id": "5e7a923dd6e4ccb8730eb95230e0c908",
        "question": "卡利創立的網際網路檔案館要到什麼時後才開放存取？",
        "paragraphs": [
            8912,
            7873,
            8209,
            7497
        ]
    },
    ...
]
```

## Training

There are two phases of the task, **context selection** and **question answering**. Two phases need to be trained seperately.

### Context Selection

```bash
python3 cs_train.py cs_config.json
```

You can also modify `cs_config.json` for different training configuration.

### Question Answering

```bash
python3 qa_train.py qa_config.json
```

You can also modify `qa_config.json` for different training configuration.

### Logging

`tensorboard` can be used for logging illustration.

```bash
tensorboard --logdir [logging directory]
```

The default logging directory is `[task]_output/runs/[current datetime]_[hostname]`.

## Testing

### Pretrained Checkpoints

Pretrained checkpoints are provided([hfl/chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large) is used for both context selection and question answering) and can be obtained by running the download script,

```bash
bash download.sh
```

### Two-phased Inference

To inference testing data with both context selection and question answering model, run

```bash
python3 inference.py inference_config.json
```

### QA-only Inference

To inference testing data with only the question answering model(concatenating all the relevant contexts as one), run

```bash
python3 inference-qa-only.py inference_config.json
```

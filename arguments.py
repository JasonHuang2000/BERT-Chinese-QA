from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    pretrained_name: str = field(
        metadata={"help": "Pretrained model name from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Use local checkpoint instead of huggingface pretrained model"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_root: str = field(
        default=None,
        metadata={"help": "Root directory where train/valid/test data are situated"}
    )
    max_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_len: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )